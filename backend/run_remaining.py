#runs the robustness benchmark for evry k thats still missing, one after the other
#uses the same pipeline_robustness_benchmark.py so settings are identical to k3 n k4
#safe to stop n rerun, the benchmark skips qustions already saved in each csv
#run from backend folder wit the server n ollama runing: python run_remaining_k.py
import subprocess,sys,time

K_VALUES=[5,2,6,1]  #k3 (ben) n k4 (me) already done
PASSES=2  #2nd pass fills in any qustions that failed the first time

for p in range(1,PASSES+1):
    for k in K_VALUES:
        print(f"\n===== pass {p} | k={k} | started {time.strftime('%H:%M')} =====",flush=True)
        r=subprocess.run([sys.executable,"pipeline_robustness_benchmark.py","--top-k",str(k)])
        if r.returncode!=0:
            print(f"k={k} stoped wit an error (code {r.returncode}), moving on to the next one",flush=True)

print(f"\nall done {time.strftime('%H:%M')}, results in ../results/robustness_benchmark")