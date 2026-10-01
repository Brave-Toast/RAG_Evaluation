"""
builds the comparsion table (RS-58) from results/comparison_runs.jsonl
    python build_comparison_table.py

writes 2 files in results/
    comparison_table.md  - readable table for the report
    comparison_table.csv - same thing for excel / charts

how to read it (answer F1 = how many words the answer shares with the ground truth, 0 to 1)
    Clean      avg F1 on the orignal question, nothing broken
    Broken     avg F1 after the pertubation with no fix
    Fixed      avg F1 after the pertubation with the fix turned on
    Recovered  how much of the lost F1 the fix got back = (Fixed - Broken) / (Clean - Broken)

stuff we leave out so the numbers are fair
    - questions where the pertubation didnt acutally change anything (nothing to fix)
    - runs where the llm failed and backend used a fake synthetic answer
    each fix is only compared to the no fix run on the smae questions so its apples to apples
"""

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict
from statistics import mean

RESULTS_DIR=pathlib.Path(__file__).resolve().parent.parent / "results"
NO_REPAIR = "none"
MIN_DROP_FOR_RECOVERY=0.02 #if the pertubation barely hurt anything then recovered % is meaningless so we skip it

def load_rows(path):
    #reads the jsonl file, each line is one saved result
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def build_table(rows,include_unapplied=False, include_synthetic=False):
    #if somthing got run twice jsut keep the latest one
    latest = {}
    for row in rows:
        latest[(row["question_id"],row["perturbation"],row["repair"])]=row

    #the no fix runs, these are what every fix gets compared agianst
    unfixed={(q,p): row for (q,p,r),row in latest.items() if r==NO_REPAIR}
    by_perturbation = defaultdict(dict) #pertubation -> fix -> list of (no fix row, fixed row) for the smae question
    skipped=defaultdict(int)

    for (q,p),base in unfixed.items():
        if not include_synthetic and base.get("synthetic"):
            skipped["synthetic answer"]+=1
            continue
        if not include_unapplied and not base.get("perturbation_applied"):
            skipped["perturbation changed nothing"] += 1
            continue
        by_perturbation[p].setdefault(NO_REPAIR,[]).append((base,base))
        #find the fixed runs for this exact question + pertubation and pair them up
        for (q2,p2,r), fixed in latest.items():
            if q2!=q or p2!=p or r==NO_REPAIR:
                continue
            if not include_synthetic and fixed.get("synthetic"):
                skipped["synthetic answer"]+=1
                continue
            by_perturbation[p].setdefault(r,[]).append((base,fixed))

    #now work out the averages, one table row per pertubation + fix
    table=[]
    for p in sorted(by_perturbation):
        pairs_none = by_perturbation[p][NO_REPAIR]
        repairs=sorted(r for r in by_perturbation[p] if r!=NO_REPAIR) or [None]
        for r in repairs:
            pairs=by_perturbation[p][r] if r else pairs_none
            clean = mean(b["answer_f1_baseline"] for b,_ in pairs)
            broken=mean(b["answer_f1_perturbed"] for b, _ in pairs)
            fixed=mean(f["answer_f1_perturbed"] for _,f in pairs) if r else None
            drop = clean-broken
            recovered=(fixed-broken)/drop if r and drop>=MIN_DROP_FOR_RECOVERY else None
            table.append({
                "perturbation":p,
                "repair":r or "(no fix yet)",
                "questions": len(pairs),
                "clean_f1":round(clean,3),
                "broken_f1":round(broken, 3),
                "fixed_f1":None if fixed is None else round(fixed,3),
                "change_from_fix":None if fixed is None else round(fixed-broken,3),
                "recovered_pct": None if recovered is None else round(100*recovered),
                "fix_triggered_pct":round(100*mean(f.get("repair_applied",False) for _,f in pairs)) if r else None,
                "context_recall_broken":round(mean(b["context_recall_perturbed"] for b,_ in pairs),3),
                "context_recall_fixed":round(mean(f["context_recall_perturbed"] for _, f in pairs),3) if r else None,
            })
    return table,dict(skipped)

def fmt(value,suffix=""):
    #shows - when theres no value insted of None
    return "-" if value is None else f"{value}{suffix}"

def write_markdown(table,skipped, path):
    #makes the readable version of the table for the report
    lines=[
        "# Perturbation x Repair Comparison",
        "",
        "Answer F1 against the ground truth (0 to 1). Recovered = share of the lost F1 the fix won back.",
        "",
        "| Perturbation | Fix | Questions | Clean | Broken | Fixed | Change | Recovered | Fix triggered |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table:
        change=row["change_from_fix"]
        lines.append(
            f"| {row['perturbation']} | {row['repair']} | {row['questions']} | {row['clean_f1']} | {row['broken_f1']} | "
            f"{fmt(row['fixed_f1'])} | {'-' if change is None else f'{change:+}'} | {fmt(row['recovered_pct'],'%')} | "
            f"{fmt(row['fix_triggered_pct'], '%')} |"
        )
    if skipped:
        lines+=["", "Left out: " + ", ".join(f"{n} run(s) ({why})" for why,n in skipped.items()) + "."]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")

def write_csv(table,path):
    #same table but as csv so it opens in excel
    with open(path,"w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

def main(argv=None):
    parser=argparse.ArgumentParser(description="Build the perturbation x repair comparison table.")
    parser.add_argument("--input",type=pathlib.Path,default=RESULTS_DIR/"comparison_runs.jsonl")
    parser.add_argument("--output-dir", type=pathlib.Path, default=RESULTS_DIR)
    parser.add_argument("--include-unapplied",action="store_true",help="keep questions the perturbation didn't change")
    parser.add_argument("--include-synthetic",action="store_true", help="keep runs that fell back to synthetic answers")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"No results at {args.input}. Run run_comparison.py first.")
        return 1
    table, skipped=build_table(load_rows(args.input),args.include_unapplied,args.include_synthetic)
    if not table:
        print("No usable rows yet (everything was filtered out). Check the runs completed without synthetic answers.")
        return 1
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_markdown(table,skipped,args.output_dir/"comparison_table.md")
    write_csv(table, args.output_dir/"comparison_table.csv")
    print((args.output_dir/"comparison_table.md").read_text(encoding="utf-8"))
    return 0

if __name__=="__main__":
    sys.exit(main())