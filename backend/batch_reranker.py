#batch test - does the reranker actualy help? runs evry qustion with reranker off n on, saves a csv + prints a summary
#uses synthetic-evaluator so no ollama needed (fast), we only care wich docs reach the llm (context recall) not the answer
#run from backend folder wit the server runing
import json,csv,statistics
import httpx

#only pre retrieval perturbs, the post retrieval ones happen after the reranker so it cant affect them
PERTURBATIONS=["spelling","emoji","negation_flip","entity_substitution","conflict","metadata_injection"]
TOP_K=2
OUT_CSV="../results/reranker_batch_bge_base.csv"

with open("hotpotqa_dataset.json",encoding="utf-8") as f:
    dataset=json.load(f)

rows=[]
for i,q in enumerate(dataset):
    if not q.get("gold_titles"):
        continue  #no gold docs = cant measure recall
    for use_reranker in [False,True]:
        payload={
            "question_id":str(q["id"]),
            "baseline_query":q["question"],
            "ground_truth":q["ground_truth"],
            "contexts":q["contexts"],
            "gold_titles":q["gold_titles"],
            "perturbation_types":PERTURBATIONS,
            "mode":"separate",               #one variant per perturbation, all share the same clean baseline
            "seed":42,
            "top_k":TOP_K,
            "llm_model":"synthetic-evaluator",
            "use_reranker":use_reranker,
        }
        try:
            res=httpx.post("http://127.0.0.1:8000/api/run-evaluation",json=payload,timeout=600)
            res.raise_for_status()
        except Exception as e:
            print(f"q{i} reranker={use_reranker} failed: {e}")
            continue
        out=res.json()
        #clean run recall, same for evry variant so jsut take it from the first
        rows.append({"q":i,"reranker":use_reranker,"perturbation":"clean",
                     "recall":out["variants"][0]["metrics"]["context_recall_baseline"],"fake_doc_reached_llm":False})
        for v in out["variants"]:
            titles=[d["title"] for d in v["perturbed_retrieved_contexts"]]
            rows.append({"q":i,"reranker":use_reranker,"perturbation":v["methods"][0],
                         "recall":v["metrics"]["context_recall_perturbed"],
                         "fake_doc_reached_llm":any("Synthetic" in t for t in titles)})  #did the conflict/metadata fake doc make it into top_k
    print(f"done {i+1}/{len(dataset)}")

with open(OUT_CSV,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

#summary - per perturbation: avg recall off vs on, how many qustions it helped/hurt, n how often fake docs got thru
print(f"\n{'perturbation':<22}{'recall off':>11}{'recall on':>11}{'helped':>8}{'hurt':>6}{'fake off':>10}{'fake on':>9}")
for p in ["clean"]+PERTURBATIONS:
    off={r["q"]:r for r in rows if r["perturbation"]==p and not r["reranker"]}
    on={r["q"]:r for r in rows if r["perturbation"]==p and r["reranker"]}
    qs=off.keys() & on.keys()
    if not qs:
        continue
    helped=sum(on[q]["recall"]>off[q]["recall"] for q in qs)
    hurt=sum(on[q]["recall"]<off[q]["recall"] for q in qs)
    fake_off=sum(off[q]["fake_doc_reached_llm"] for q in qs)
    fake_on=sum(on[q]["fake_doc_reached_llm"] for q in qs)
    print(f"{p:<22}{statistics.mean(off[q]['recall'] for q in qs):>11.3f}{statistics.mean(on[q]['recall'] for q in qs):>11.3f}"
          f"{helped:>8}{hurt:>6}{fake_off:>10}{fake_on:>9}")
print(f"\nquestions: {len({r['q'] for r in rows})} | csv saved to {OUT_CSV}")