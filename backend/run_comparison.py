"""
runner for the comparsion table (RS-58)
basicaly does what we do on the dashbaord but automaticaly, runs every pertubation with and
without its fix thru the backend and saves each result as one line in a jsonl file

backend needs ot be running first (python app.py) then from the backend folder:
    python run_comparison.py --dry-run         #jsut prints the plan and how many llm calls it wil take
    python run_comparison.py --questions 10    #acutally runs it
    python build_comparison_table.py           #turns the results into the table

fixes named fix_<pertubation> like fix_spelling get tested on that pertubation automaticaly
for any other name add it to DEFAULT_PAIRS below or pass it in like --pair spelling=spell_check
--matrix tests evry fix on evry pertubation (takes way longer)

smae seed is used for a question with and without the fix so the broken qeustion is exactly
the same both times, only diffrence is the fix. if it crashes jsut run it agian it skips
whatevr is allready saved
"""

import argparse
import datetime
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

#remember to match the pertubations in the app.py and here as well. 
ALL_PERTURBATIONS = [
    "spelling","rephrase","emoji","negation_flip", "entity_substitution","conflict",
    "remove_evidence", "alter_facts","truncate","irrelevant", "shuffle",
]
#fixes that dont follow the fix_<pertubation> naming, paird with what there suposed to fix
DEFAULT_PAIRS=[("conflict","contradiction_filter")] #pertubation names comes firs, then the repair name given to it 
MAX_METHODS_PER_REQUEST = 12 #has to match max_length in app.py, backend only lets this many pertubations thru per request
NO_REPAIR="none"
DEFAULT_OUTPUT=pathlib.Path(__file__).resolve().parent.parent / "results"/"comparison_runs.jsonl"

def http_json(url, payload=None,timeout=60):
    #sends a reqeust to the backend and gives back the json it replys with
    #no payload means GET, with payload its a POST, same thing ragEngine.js does on the frontend
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request=urllib.request.Request(
        url,data=data, method="GET" if payload is None else "POST",
        headers={"Content-Type":"application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request,timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def fetch_repairs(backend):
    #asks /api/repairs which fixes are registerd in repairs.py
    try:
        return [r["name"] for r in http_json(f"{backend}/api/repairs").get("repairs",[])]
    except urllib.error.HTTPError as exc:
        if exc.code==404:
            print("Note: backend has no /api/repairs endpoint yet, so no repairs will be tested.")
            return []
        raise

def build_pairs(perturbations,repairs, manual_pairs,matrix):
    #figures out wich fixes to test on wich pertubation
    #ends up lookin like {"spelling":["fix_spelling"],"conflict":["contradiction_filter"],"emoji":[]}
    pairs = {p:[] for p in perturbations}
    for p in perturbations:
        if matrix:
            pairs[p]=list(repairs)
        elif f"fix_{p}" in repairs:
            pairs[p].append(f"fix_{p}")
    for p,r in DEFAULT_PAIRS:
        if p in pairs and r in repairs and r not in pairs[p]:
            pairs[p].append(r)
    for p, r in manual_pairs:
        if p in pairs and r not in pairs[p]:
            pairs[p].append(r)
    return pairs

def plan_requests(questions,pairs):
    #makes the list of reqeusts we need to send
    #evry pertubation runs once with no fix first so we can see how bad it breaks stuff
    #then each fix only runs on the pertubations its paird with
    #uses separate mode so evry pertubation starts from the clean question not stacked
    conditions={NO_REPAIR: [p for p in pairs]}
    for p,repair_names in pairs.items():
        for r in repair_names:
            conditions.setdefault(r,[]).append(p)
    plan = []
    for q_index,question in enumerate(questions):
        for repair, perts in conditions.items():
            for start in range(0,len(perts),MAX_METHODS_PER_REQUEST):
                plan.append((q_index,question,repair,perts[start:start + MAX_METHODS_PER_REQUEST]))
    return plan

def load_done(output):
    #reads the results file and rembers what we allready ran so we dont redo it
    done=set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((str(row["question_id"]),row["perturbation"], row["repair"]))
    return done

def rows_from_response(response,question,repair,seed,args):
    #takes the backend reply and pulls out the stuff we need, one row per pertubation
    #perturbation_applied = did the pertubation acutally change anything or not
    #repair_applied = did the fix do somthing or did it jsut skip
    #synthetic = llm failed and backend gave a fake answer instead, we dont want thoes in the table
    #failed_at = swap tset result, wich stage broke teh anwser (QUERY/RETRIEVAL/CONTEXT/GENERATION)
    rows=[]
    for variant in response.get("variants", []):
        perturbation=variant["methods"][0]
        changes = variant.get("changes",[])
        repair_changes=[c for c in changes if c.get("method") not in ALL_PERTURBATIONS]
        m=variant["metrics"]
        loc=variant.get("failure_localisation") or {}
        rows.append({
            "timestamp":datetime.datetime.now().isoformat(timespec="seconds"),
            "question_id": str(question["id"]),
            "question":question["question"],
            "perturbation":perturbation,
            "repair": repair,
            "seed":seed,
            "llm_model":args.llm_model,
            "embedding_provider": args.embedding_provider,
            "top_k":args.top_k,
            "perturbation_applied":any(c.get("method")==perturbation and c.get("status") == "applied" for c in changes),
            "repair_applied":any(c.get("status")=="applied" for c in repair_changes),
            "repair_changes":repair_changes,
            "answer_f1_baseline": m.get("answer_f1_baseline"),
            "answer_f1_perturbed":m.get("answer_f1_perturbed"),
            "context_recall_baseline":m.get("context_recall_baseline"),
            "context_recall_perturbed": m.get("context_recall_perturbed"),
            "answer_vs_truth_baseline":m.get("generation_vs_gt_baseline"),
            "answer_vs_truth_perturbed":m.get("generation_vs_gt_perturbed"),
            "failure_attribution":m.get("failure_attribution"),
            "failure_root_cause": m.get("failure_root_cause"),
            "failed_at":loc.get("failed_at"),
            "failed_at_reason":loc.get("reason"),
            "baseline_source":variant.get("baseline_generation_source"),
            "perturbed_source":variant.get("perturbed_generation_source"),
            "synthetic":"synthetic" in (variant.get("baseline_generation_source"),variant.get("perturbed_generation_source")),
            "perturbed_query":variant.get("perturbed_query"),
            "ground_truth": variant.get("ground_truth"),
            "baseline_answer":variant.get("baseline_answer"),
            "perturbed_answer":variant.get("perturbed_answer"),
            "warning":variant.get("warning"),
        })
    return rows

def parse_pair(text):
    #turns spelling=spell_check from the terminal into ("spelling","spell_check")
    if "=" not in text:
        raise argparse.ArgumentTypeError("use perturbation=repair_name, e.g. spelling=spell_check")
    p, r=(part.strip() for part in text.split("=",1))
    if p not in ALL_PERTURBATIONS:
        raise argparse.ArgumentTypeError(f"unknown perturbation '{p}'")
    return p,r

def main(argv=None):
    #all the options u can pass in from the terminal
    parser = argparse.ArgumentParser(description="Run perturbation x repair comparisons through the backend.")
    parser.add_argument("--backend",default="http://127.0.0.1:8000")
    parser.add_argument("--questions", type=int, default=10, help="how many benchmark questions to use")
    parser.add_argument("--offset",type=int,default=0,help="skip this many questions first")
    parser.add_argument("--perturbations",nargs="+",default=ALL_PERTURBATIONS, choices=ALL_PERTURBATIONS)
    parser.add_argument("--pair",action="append",default=[],type=parse_pair,metavar="PERTURBATION=REPAIR")
    parser.add_argument("--matrix", action="store_true", help="test every repair on every perturbation")
    parser.add_argument("--seed",type=int,default=1000,help="question i uses seed + i")
    parser.add_argument("--llm-model",default="ollama/qwen3.5:9b")
    parser.add_argument("--embedding-provider", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--search-type",default="similarity",choices=["similarity","mmr"])
    parser.add_argument("--top-k",type=int, default=2)
    parser.add_argument("--typo-rate",type=float,default=0.25)
    parser.add_argument("--timeout",type=int,default=1800,help="seconds allowed per request")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run",action="store_true",help="print the plan without running anything")
    args=parser.parse_args(argv)

    # grab the qeustions and list of fixes from the backend, stop here if its not running
    backend = args.backend.rstrip("/")
    try:
        dataset=http_json(f"{backend}/api/dataset")["dataset"]
        repairs = fetch_repairs(backend)
    except (urllib.error.URLError,OSError) as exc:
        print(f"Can't reach the backend at {backend} ({exc}). Start it with: python app.py")
        return 1

    questions=dataset[args.offset:args.offset+args.questions]
    unknown = sorted({r for _,r in args.pair}-set(repairs))
    if unknown:
        print(f"Warning: these paired repairs aren't registered in repairs.py: {', '.join(unknown)}")
    pairs=build_pairs(args.perturbations,repairs,args.pair,args.matrix)
    plan = plan_requests(questions,pairs)
    done=load_done(args.output)

    #print the plan first so we can chek it b4 anything runs
    print(f"Questions: {len(questions)}  |  Registered repairs: {', '.join(repairs) or 'none'}")
    for p,rs in pairs.items():
        print(f"  {p:<20} tested with: none{''.join(', ' + r for r in rs)}")
    todo=[]
    for q_index,question,repair,perts in plan:
        remaining = [p for p in perts if (str(question["id"]),p,repair) not in done]
        if remaining:
            todo.append((q_index,question,repair,remaining))
    generations=sum(1+len(perts) for *_, perts in todo)
    print(f"Requests to run: {len(todo)} ({len(plan)-len(todo)} already done)  |  LLM answer generations: ~{generations}")
    if args.dry_run or not todo:
        return 0

    #acutally run evrything, saves after each reqeust so nothing gets lost if it crashes halfway
    args.output.parent.mkdir(parents=True,exist_ok=True)
    failures, started=0,time.time()
    for n,(q_index,question,repair,perts) in enumerate(todo,1):
        seed = args.seed+args.offset+q_index #smae seed evry time for this question = smae broken input
        payload={
            "question_id":question["id"],
            "baseline_query": question["question"],
            "ground_truth":question["ground_truth"],
            "contexts":question["contexts"],
            "gold_titles":question.get("gold_titles") or [],
            "supporting_titles": question.get("supporting_titles"),
            "perturbation_types":perts,
            "mode":"separate",
            "seed":seed,
            "repair_methods":None if repair==NO_REPAIR else [repair],
            "typo_rate": args.typo_rate,
            "embedding_provider":args.embedding_provider,
            "search_type":args.search_type,
            "top_k": args.top_k,
            "llm_model":args.llm_model,
        }
        t0=time.time()
        print(f"[{n}/{len(todo)}] question {q_index+args.offset+1} | repair={repair} | {', '.join(perts)} ... ", end="",flush=True)
        try:
            response = http_json(f"{backend}/api/run-evaluation",payload,timeout=args.timeout)
        except urllib.error.HTTPError as exc:
            failures+=1
            print(f"failed (HTTP {exc.code}): {exc.read().decode('utf-8','replace')[:300]}")
            continue
        except (urllib.error.URLError, OSError) as exc:
            failures += 1
            print(f"failed: {exc}")
            continue
        rows=rows_from_response(response,question,repair,seed,args)
        with open(args.output,"a",encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False)+"\n")
        synthetic=sum(r["synthetic"] for r in rows)
        print(f"{time.time()-t0:.0f}s" + (f"  (warning: {synthetic} synthetic fallback answers)" if synthetic else ""))

    elapsed = time.time()-started
    print(f"\nDone in {elapsed/60:.1f} min, about {elapsed/max(generations,1):.1f}s per LLM answer. "
          f"{failures} failed request(s). Results: {args.output}")
    if failures:
        print("Re-run the same command to retry the failed ones.")
    return 0

if __name__=="__main__":
    sys.exit(main())