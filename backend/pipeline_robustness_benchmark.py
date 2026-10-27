#full pipeline robustness test - perturbation -> retrieval -> bge reranker -> llm answer -> llm judge
#evry team member runs this once wit there own k value (4 to 9), all 12 perturbations, same 40 qustions
#results file is named from the k + llm automaticaly (eg k4_qwen.csv) so nobody changes anything
#saves after evry qustion, if it crashes or u stop it just run the same command again n it carries on
#run from backend folder wit the server n ollama runing
#  quick test:        python pipeline_robustness_benchmark.py --top-k 4 --n-questions 3
#  real run:          python pipeline_robustness_benchmark.py --top-k 4
#  extra (optional):  python pipeline_robustness_benchmark.py --top-k 4 --llm mistral
#combining evryones results is in combine_robustness_results.py
import json,csv,statistics,argparse,os,subprocess
import httpx

SERVER="http://127.0.0.1:8000/api/run-evaluation"
EMBEDDING="sentence-transformers/all-mpnet-base-v2"
RERANKER="BAAI/bge-reranker-base"
JUDGE_MODEL="ollama/qwen3.5:9b"
FETCH_K=10
SEED=42
N_QUESTIONS=40  #same first 40 for evryone
CORRECT=0.5  #same cutof the pipeline uses for f1 n judge correctness
LLMS={"qwen":"ollama/qwen3.5:9b","mistral":"ollama/mistral"}
#pre = damages the query (b4 retrieval), post = damages the docs the llm gets (after retrieval)
PRE_RETRIEVAL=["spelling","emoji","rephrase","negation_flip","entity_substitution","conflict","metadata_injection"]
POST_RETRIEVAL=["remove_evidence","alter_facts","truncate","irrelevant","shuffle"]
PERTURBATIONS=PRE_RETRIEVAL+POST_RETRIEVAL
OUT_DIR="../results/robustness_benchmark"
TEST_DIR="../results/robustness_benchmark_test"  #test runs go here so they dont mix wit real results
FIELDS=["top_k","llm","q","question_id","question","ground_truth","perturbation","stage","recall",
        "answer_f1","f1_correct","judge_correctness","judge_faithfulness","judge_verdict","judge_correct",
        "scorers_disagree","attribution_f1","attribution_judge","fake_doc_reached_llm","suspicious",
        "commit","model_id","answer"]

ap=argparse.ArgumentParser()
ap.add_argument("--top-k",type=int,required=True,help="ur assigned k value")
ap.add_argument("--llm",default="qwen",choices=list(LLMS))
ap.add_argument("--n-questions",type=int,default=None,help="only for quick tests")
args=ap.parse_args()

#---- helpers ----
def git_commit():
    try:
        return subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"

def ollama_id(llm):
    #model id from ollama list so we can tell if someone has a difrent version of the model
    name=llm.replace("ollama/","")
    try:
        out=subprocess.run(["ollama","list"],capture_output=True,text=True).stdout
    except Exception:
        return "unknown"
    for line in out.splitlines()[1:]:
        parts=line.split()
        if parts and parts[0] in (name,name+":latest"):
            return parts[1]
    return "not installed"

def pick(v,key,default=None):
    #some fields sit in the variant, some inside metrics, so look in both
    if key in v:
        return v[key]
    return v.get("metrics",{}).get(key,default)

def looks_broken(ans):
    #empty or error answers = llm call probly failed, dont trust that row
    a=(ans or "").strip().lower()
    return a=="" or a.startswith("error") or "failed" in a[:40]

def verdict_from(retrieval_ok,answer_ok):
    #same 4 boxes as diagnose_failure
    if retrieval_ok and answer_ok: return "SUCCESS"
    if not retrieval_ok and not answer_ok: return "RETRIEVAL_FAILURE"
    if retrieval_ok: return "GENERATION_FAILURE"
    return "PARAMETRIC_LEAKAGE"

def make_row(base,perturbation,recall,n_gold,ans,f1,judge_corr,judge_faith,judge_verdict,fake):
    retrieval_ok=recall>=(1.0 if n_gold<=1 else 0.5)  #same rule as diagnose_failure
    f1_ok=f1>=CORRECT
    judge_ok=None if judge_corr is None else judge_corr>=CORRECT
    return {**base,"perturbation":perturbation,
            "stage":"clean" if perturbation=="clean" else ("pre" if perturbation in PRE_RETRIEVAL else "post"),
            "recall":recall,"answer_f1":f1,"f1_correct":f1_ok,
            "judge_correctness":"" if judge_corr is None else judge_corr,
            "judge_faithfulness":"" if judge_faith is None else judge_faith,
            "judge_verdict":judge_verdict or "",
            "judge_correct":"" if judge_ok is None else judge_ok,
            "scorers_disagree":judge_ok is not None and judge_ok!=f1_ok,
            "attribution_f1":verdict_from(retrieval_ok,f1_ok),
            "attribution_judge":"" if judge_ok is None else verdict_from(retrieval_ok,judge_ok),
            "fake_doc_reached_llm":fake,"suspicious":looks_broken(ans),"answer":ans}

def to_bool(x):
    return None if x=="" else x=="True"

def load_rows(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["recall"]=float(r["recall"])
            for c in ["f1_correct","judge_correct","scorers_disagree","fake_doc_reached_llm","suspicious"]:
                r[c]=to_bool(r[c])
            rows.append(r)
    return rows

def mean(xs):
    xs=[x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None

def pct(x):
    return f"{100*x:.1f}%" if x is not None else "-"

#---- setup ----
k=args.top_k
if k>FETCH_K:
    raise SystemExit(f"top_k cant be bigger then fetch_k ({FETCH_K})")
llm=LLMS[args.llm]
for m in {llm,JUDGE_MODEL}:
    if ollama_id(m)=="not installed":
        raise SystemExit(f"{m} not installed, run: ollama pull {m.replace('ollama/','')}")

out_dir=TEST_DIR if args.n_questions else OUT_DIR
os.makedirs(out_dir,exist_ok=True)
path=os.path.join(out_dir,f"k{k}_{args.llm}.csv")  #file name comes from k + llm automaticaly

with open("hotpotqa_dataset.json",encoding="utf-8") as f:
    dataset=[q for q in json.load(f) if q.get("gold_titles")][:args.n_questions or N_QUESTIONS]  #no gold docs = cant measure recall

#carry on from where it stoped if the file already has some qustions in it
done=set()
if os.path.exists(path):
    with open(path,encoding="utf-8") as f:
        done={r["question_id"] for r in csv.DictReader(f)}
new_file=not os.path.exists(path)

meta={"commit":git_commit(),"model_id":ollama_id(llm)}
print(f"top_k {k} | llm {args.llm} | {len(dataset)} qustions ({len(done)} already done) | 12 perturbations | judge on | saving to {path}")

#---- the run ----
checked=False
with open(path,"a",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=FIELDS)
    if new_file:
        w.writeheader()
    for i,q in enumerate(dataset):
        if str(q["id"]) in done:
            continue
        payload={
            "question_id":str(q["id"]),
            "baseline_query":q["question"],
            "ground_truth":q["ground_truth"],
            "contexts":q["contexts"],
            "gold_titles":q["gold_titles"],
            "perturbation_types":PERTURBATIONS,
            "mode":"separate",               #one variant per perturbation, all share the same clean baseline
            "seed":SEED,
            "top_k":k,
            "llm_model":llm,
            "embedding_provider":EMBEDDING,
            "use_reranker":True,
            "reranker_model":RERANKER,
            "rerank_fetch_k":FETCH_K,
            "use_llm_judge":True,
            "judge_model":JUDGE_MODEL,
        }
        try:
            res=httpx.post(SERVER,json=payload,timeout=3600)  #llm + judge calls can be slow on some laptops
            res.raise_for_status()
        except Exception as e:
            print(f"  q {i+1} failed: {e} (will be retried next time u run it)")
            continue
        out=res.json()

        #first qustion only: make sure the embedding n the judge actualy worked
        if not checked:
            used=out.get("embedding_model")
            if used and used!=EMBEDDING:
                raise SystemExit(f"server used {used} insted of {EMBEDDING}, stoping")
            if all(pick(v,"judge_correctness_perturbed") is None for v in out["variants"]):
                raise SystemExit("judge didnt return any scores, chek the judge is working b4 runing this")
            checked=True

        base={**meta,"top_k":k,"llm":args.llm,"q":i,"question_id":str(q["id"]),
              "question":q["question"],"ground_truth":q["ground_truth"]}
        n_gold=len(q["gold_titles"])
        v0=out["variants"][0]
        rows=[make_row(base,"clean",pick(v0,"context_recall_baseline"),n_gold,
                       pick(v0,"baseline_answer","") or "",pick(v0,"answer_f1_baseline",0.0),
                       pick(v0,"judge_correctness_baseline"),pick(v0,"judge_faithfulness_baseline"),
                       pick(v0,"judge_verdict_baseline"),False)]
        for v in out["variants"]:
            titles=[d["title"] for d in v.get("perturbed_retrieved_contexts",[])]
            rows.append(make_row(base,v["methods"][0],pick(v,"context_recall_perturbed"),n_gold,
                                 pick(v,"perturbed_answer","") or "",pick(v,"answer_f1_perturbed",0.0),
                                 pick(v,"judge_correctness_perturbed"),pick(v,"judge_faithfulness_perturbed"),
                                 pick(v,"judge_verdict_perturbed"),
                                 any("Synthetic" in t for t in titles)))  #did the fake doc make it into top_k
        w.writerows(rows)
        f.flush()  #saved after evry qustion so a crash only loses the one its on
        print(f"  q {i+1}/{len(dataset)} done")

#---- summary for ur insight ----
rows=load_rows(path)
missing=len(dataset)-len({r["question_id"] for r in rows})
clean=[r for r in rows if r["perturbation"]=="clean"]
clean_judge=mean(r["judge_correct"] for r in clean)

print(f"\n== k={k} {args.llm} ==")
print(f"{'perturbation':<22}{'stage':>6}{'recall':>8}{'f1 acc':>9}{'judge acc':>11}{'judge drop':>12}{'disagree':>10}{'fake doc':>10}")
for p in ["clean"]+PERTURBATIONS:
    rs=[r for r in rows if r["perturbation"]==p]
    if not rs:
        continue
    judge_acc=mean(r["judge_correct"] for r in rs)
    drop=clean_judge-judge_acc if clean_judge is not None and judge_acc is not None else None
    print(f"{p:<22}{rs[0]['stage']:>6}{mean(r['recall'] for r in rs):>8.3f}{pct(mean(r['f1_correct'] for r in rs)):>9}"
          f"{pct(judge_acc):>11}{pct(drop):>12}{pct(mean(r['scorers_disagree'] for r in rs)):>10}"
          f"{pct(mean(r['fake_doc_reached_llm'] for r in rs)):>10}")

#query damage vs doc damage
for stage,label in [("pre","damaged query (pre retrieval)"),("post","damaged docs (post retrieval)")]:
    acc=mean(r["judge_correct"] for r in rows if r["stage"]==stage)
    if acc is not None and clean_judge is not None:
        print(f"\n{label}: avg judge accuracy {pct(acc)}, drop vs clean {pct(clean_judge-acc)}")

both=[r for r in rows if r["judge_correct"] is not None and not r["suspicious"]]
if both:
    f1_only=sum(1 for r in both if r["f1_correct"] and not r["judge_correct"])
    judge_only=sum(1 for r in both if r["judge_correct"] and not r["f1_correct"])
    print(f"\nf1 vs judge: disagree on {pct((f1_only+judge_only)/len(both))} of answers")
    print(f"  f1 says rigth, judge says wrong: {f1_only}")
    print(f"  judge says rigth, f1 says wrong: {judge_only}")

bad=sum(bool(r["suspicious"]) for r in rows)
if bad:
    print(f"\nWARNING {bad} answers look broken (empty/error), chek the csv b4 using these results")
if missing:
    print(f"\n{missing} qustions missing (failed), run the same command again to fill them in")
else:
    print(f"\nall done, push {path} n use these tables for ur insight paragraph")