# this assumes all the correct {run}_phi_theta.csv files exist in the correct folder

def run_cmd(cmd, shell=True, cwd=None):
    print(f"\n--- Running command: {cmd} ---")
    result = subprocess.run(cmd, shell=shell, cwd=cwd)
    if result.returncode != 0:
        print(f"!!! Command failed: {cmd}")
    print(f"--- Finished command: {cmd} ---\n")
    return result.returncode

runlist = np.loadtxt('bronze_runlist_18runs.txt', dtype=int)
get_runlist_stats_path= '/home/claramariadima/SNO/RS_isotropy/get_runlist_stats'

    # 10. Final step: run compute_run_metrics.py.
for run in runlist:
    compute_stats_cmd = f"python3 compute_run_metrics.py -run_number {run}"
    run_cmd(compute_stats_cmd, cwd=get_runlist_stats_path)
