#!/usr/bin/env python
import os
import sys
import subprocess
import numpy as np
import datetime
import time
import random
import string
import re
import glob
import concurrent.futures
import argparse

# === Set up argument parser ===

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download, process runs, and compute coverage metrics"
    )
    parser.add_argument(
        "--runlist-filename",
        type=str,
        default="bronze_runlist_6runs.txt",
        help="Path to text file containing the list of run numbers"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="random_coverage_stats.csv",
        help="Output CSV file for coverage and isotropy metrics"
    )
    parser.add_argument(
        "--output-pmtmap-dir",
        type=str,
        default="random_list",
        help="Directory name under PMT_maps where PDFs will be saved"
    )
    parser.add_argument(
        "--download-streams",
        type=int,
        default=6,
        help="Maximum simultaneous download threads"
    )
    parser.add_argument(
        "--proc-streams",
        type=int,
        default=6,
        help="Maximum simultaneous processing threads"
    )
    
    # To run this script with correct arguments (example arguments for testing given, can change for others):
    # python3 full_loop_optimized.py --runlist-filename "bronze_runlist_6runs.txt" --output-csv "random_coverage_stats.csv" --output-pmtmap-dir "random_list" --download-streams 6 --proc-streams 6
    
    return parser.parse_args()

# === Setup logging ===

def random_log_name():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"log_{ts}_{rand}.txt"

class Logger:
    def __init__(self, stream, logfile):
        self.terminal = stream
        self.logfile = logfile
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

# === Helper to run shell commands ===

def run_cmd(cmd, shell=True, cwd=None, capture_output=False):
    print(f"\n--- Running command: {cmd} ---")
    result = subprocess.run(cmd, shell=shell, cwd=cwd, capture_output=capture_output, text=True)
    if result.returncode != 0:
        print(f"!!! Command failed: {cmd}")
    print(f"--- Finished command: {cmd} ---\n")
    return result

###############################################################################
# Function to download one run's zdab using a separate filelist.
###############################################################################

def download_run(run, grid_singularity_path, grid_tools_path, grid_downloaded_path,
                 couch_user, couch_pass, singularity_image_path):
    delay = random.uniform(1, 6)
    print(f"Run {run}: Sleeping {delay:.1f}s before raw_list.")
    time.sleep(delay)
    print(f"\n=== Starting download for run {run} ===")

    # Prepare raw_list expect script
    temp_filelist = f"filelist_{run}.dat"
    expect_raw = os.path.join(grid_singularity_path, f"run_expect_{run}.exp")
    raw_cmd = f"python3 raw_list -type L1 -r {run} {run+1} -o {temp_filelist}"
    singularity_raw = (
        f"singularity exec -B /run/shm {singularity_image_path} /bin/bash -c "
        f"\"export MYPROXY_SERVER=myproxy.gridpp.rl.ac.uk && "
        f"export LCG_GFAL_INFOSYS=topbdii.grid.help.ph.ic.ac.uk:2170,lcgbdii.gridpp.rl.ac.uk:2170 && "
        f"cd {grid_tools_path} && {raw_cmd}\""
    )
    with open(expect_raw, 'w') as f:
        f.write(
            f"""#!/usr/bin/expect -f
set timeout -1
spawn {singularity_raw}
expect \"Username:\"
send \"{couch_user}\r\"
expect \"Password:\"
send \"{couch_pass}\r\"
expect eof
"""
        )
    os.chmod(expect_raw, 0o755)
    run_cmd(f"expect {expect_raw}", cwd=grid_tools_path)

    # Modify filelist
    fl_path = os.path.join(grid_tools_path, temp_filelist)
    with open(fl_path) as f:
        lines = f.readlines()
    if lines and len(lines[0].split("\t")) >= 3:
        parts = lines[0].strip().split("\t")
        parts[2] = re.sub(
            r".*/l1",
            "srm://lcg-snopse1.sfu.computecanada.ca:8443/snoplus/snotflow/raw/l1",
            parts[2]
        )
        with open(fl_path, 'w') as f:
            f.write("\t".join(parts) + "\n")
        print(f"filelist_{run}.dat updated.")
    else:
        print(f"⚠️ Unexpected or empty filelist for run {run}")

    # Grabber expect script and retry
    expect_gb = os.path.join(grid_singularity_path, f"run_grabber_{run}.exp")
    grab_cmd = f"python3 grabber -l {temp_filelist} --streams 1 --timeout 144000"
    singularity_gb = (
        f"singularity exec -B /run/shm {singularity_image_path} /bin/bash -c "
        f"\"export MYPROXY_SERVER=myproxy.gridpp.rl.ac.uk && "
        f"export LCG_GFAL_INFOSYS=topbdii.grid.help.ph.ic.ac.uk:2170,lcgbdii.gridpp.rl.ac.uk:2170 && "
        f"cd {grid_tools_path} && {grab_cmd}\""
    )
    with open(expect_gb, 'w') as f:
        f.write(
            f"""#!/usr/bin/expect -f
set timeout -1
spawn {singularity_gb}
expect \"download\"
send \"y\r\"
expect eof
"""
        )
    os.chmod(expect_gb, 0o755)

    retry = 0
    while True:
        print(f"Running grabber attempt {retry+1} for {run}")
        res = subprocess.run(f"expect {expect_gb}", shell=True,
                             cwd=grid_tools_path, capture_output=True, text=True)
        out = res.stdout + res.stderr
        if 'Connection timed out' in out:
            retry += 1
            print(f"Timeout, retry {retry}... sleeping 5s")
            time.sleep(5)
            continue
        if res.returncode != 0:
            raise RuntimeError(f"Grabber failed for run {run}: {out}")
        break

    # Wait for .zdab file
    zdab = f"SNOP_0000{run}_000.zdab"
    zdab_p = os.path.join(grid_downloaded_path, zdab)
    start = time.time()
    while True:
        if os.path.exists(zdab_p) and os.path.getsize(zdab_p) > 0:
            time.sleep(5)
            if os.path.getsize(zdab_p) == os.path.getsize(zdab_p):
                print(f"{zdab} fully downloaded.")
                break
        if time.time() - start > 14400:
            raise TimeoutError(f"Timeout waiting for {zdab}")
        time.sleep(5)

    # Cleanup temp scripts and filelist
    for path in (expect_raw, expect_gb, fl_path):
        try: os.remove(path)
        except: pass

    print(f"=== Download completed for run {run} ===")
    return run

###############################################################################
# Function to process one run's zdab: move, container build & run, cleanup.
###############################################################################

def process_run(run, grid_singularity_path, grid_tools_path, grid_downloaded_path,
                phi_theta_path, get_runlist_stats_path, rat_path,
                output_csv, output_pmtmap_dir):
    print(f"\n=== Starting RAT processing for run {run} ===")

    # Ensure correct branch
    desired_branch = "get_pmt_plots"
    res_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=rat_path, capture_output=True, text=True
    )
    current_branch = res_branch.stdout.strip()
    if current_branch != desired_branch:
        print(f"Switching RAT repo from '{current_branch}' to '{desired_branch}'")
        subprocess.run(["git", "checkout", desired_branch], cwd=rat_path)

    # Move zdab into RAT repo
    zdab = f"SNOP_0000{run}_000.zdab"
    src = os.path.join(grid_downloaded_path, zdab)
    dst = os.path.join(rat_path, zdab)
    if os.path.exists(src):
        os.rename(src, dst)
        print(f"Moved {zdab} to RAT for run {run}")
        # remove leftover filelist
        fl = os.path.join(grid_tools_path, f"filelist_{run}.dat")
        if os.path.exists(fl): os.remove(fl)
    else:
        print(f"⚠️ zdab not found: {src}")

    # Create macro
    mac_in = os.path.join(rat_path, "get_pmt_positions.mac")
    mac_out = os.path.join(rat_path, f"get_pmt_positions_{run}.mac")
    with open(mac_in) as fin, open(mac_out, 'w') as fout:
        for line in fin:
            if line.startswith("/rat/inzdab/load"):
                fout.write(f"/rat/inzdab/load {zdab}\n")
            else:
                fout.write(line)
    print(f"Macro written: {mac_out}")

    # Build & run docker
    script = os.path.join(rat_path, f"run_rat_macro_{run}.sh")
    with open(script, 'w') as f:
        f.write("#!/bin/sh\ncd /rat\nscons -j8\n")
        f.write(f"rat get_pmt_positions_{run}.mac\n")
    os.chmod(script, 0o755)

    docker_cmd = (
        f"sudo -S docker run --init --rm "
        f"-v {rat_path}:/rat -w /rat "
        f"snoplus/rat-container:root6 ./run_rat_macro_{run}.sh"
    )
    # Feed sudo_pass from global
    proc = subprocess.Popen(docker_cmd, shell=True,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)
    proc.stdin.write(sudo_pass + "\n")
    proc.stdin.flush()
    for line in proc.stdout: print(line, end='')
    proc.wait()
    if proc.returncode != 0:
        print(f"❌ Docker error code {proc.returncode}")
    else:
        print(f"✅ Docker succeeded for run {run}")

    # Cleanup macro & script
    for p in (mac_out, script):
        try: os.remove(p)
        except: pass

    # Post-docker cleanup & move
    if os.path.exists(dst): os.remove(dst)

    phi_fn = f"{run}_phi_theta.csv"
    phi_src = os.path.join(rat_path, phi_fn)
    phi_dst = os.path.join(phi_theta_path, phi_fn)
    if os.path.exists(phi_src): os.rename(phi_src, phi_dst)

    idx = os.path.join(rat_path, f"{run}_indices.txt")
    if os.path.exists(idx): os.remove(idx)

    for lf in glob.glob(os.path.join(rat_path, "rat.*.log")): os.remove(lf)

    for ext in ("root", "ratdb"):
        dq = os.path.join(rat_path, f"DATAQUALITY_RECORDS_{run}_p1.{ext}")
        if os.path.exists(dq): os.remove(dq)

    for tag in ("applied", "flags"):
        fpath = os.path.join(rat_path, f"{tag}_{run}_p1.png")
        if os.path.exists(fpath): os.remove(fpath)

    for name in ['PMT coverage map before PMT Calibrations.png', 'PMT coverage map after PMT Calibrations.png']:
        pth = os.path.join(rat_path, name)
        if os.path.exists(pth): os.remove(pth)

    for fn in ("TGraph_GeoCoverageMap.png", f"TGraph_GeoCoverageMap_{run}_p1.png"):
        pth = os.path.join(rat_path, fn)
        if os.path.exists(pth): os.remove(pth)

    for fn in (f"TH2D_CrateCoverageMap_{run}_p1.png", f"TH2D_UnCalCrateCoverageMap_{run}_p1.png"):
        pth = os.path.join(rat_path, fn)
        if os.path.exists(pth): os.remove(pth)

    # Invoke metrics script
    metrics_cmd = (
        f"python3 compute_run_metrics_custom_output.py "
        f"--run-number {run} --output-csv {output_csv} "
        f"--output-pmtmap-dir {output_pmtmap_dir}"
    )
    run_cmd(metrics_cmd, cwd=get_runlist_stats_path)
    print(f"=== Processing completed for run {run} ===")
    return run

###############################################################################
# Main flow
###############################################################################
if __name__ == "__main__":
    args = parse_args()

    # Load runlist and parameters
    runlist = np.loadtxt(args.runlist_filename, dtype=int)
    output_csv = args.output_csv
    output_pmtmap_dir = args.output_pmtmap_dir
    MAX_DOWNLOADS = args.download_streams
    MAX_PROCS = args.proc_streams

    # Setup logging
    log_file = open(random_log_name(), 'w')
    sys.stdout = Logger(sys.stdout, log_file)
    sys.stderr = Logger(sys.stderr, log_file)

    # Paths & creds
    grid_singularity_path = '/home/claramariadima/SNO/Singularititty'
    grid_tools_path       = '/home/claramariadima/SNO/rat-tools/GridTools'
    grid_downloaded_path  = '/home/claramariadima/SNO/rat-tools/GridTools/downloaded'
    phi_theta_path        = '/home/claramariadima/SNO/RS_isotropy/get_runlist_stats/phi_theta'
    get_runlist_stats_path= '/home/claramariadima/SNO/RS_isotropy/get_runlist_stats'
    rat_path              = '/home/claramariadima/SNO/rat'
    couch_user            = 'snoplus'
    couch_pass            = 'BiPo214intheneck'
    sudo_pass             = ':18kyorosteaua18:'
    singularity_image_path= os.path.join(grid_singularity_path, 'snopl7.img')

    print(f"\n--- Pipeline start for {len(runlist)} runs ---\n")

    dl_exec = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_DOWNLOADS)
    pr_exec = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PROCS)

    # Schedule downloads
    future_to_run = {}
    for run in runlist:
        fut = dl_exec.submit(
            download_run, run,
            grid_singularity_path, grid_tools_path,
            grid_downloaded_path, couch_user,
            couch_pass, singularity_image_path
        )
        future_to_run[fut] = run

    # Chain processing
    for fut in concurrent.futures.as_completed(future_to_run):
        run = fut.result()
        print(f"\n>> Download done for {run}; queueing processing...\n")
        pr_exec.submit(
            process_run, run,
            grid_singularity_path, grid_tools_path,
            grid_downloaded_path, phi_theta_path,
            get_runlist_stats_path, rat_path,
            output_csv, output_pmtmap_dir
        )

    dl_exec.shutdown(wait=True)
    pr_exec.shutdown(wait=True)

    log_file.close()
    print(f"\nAll done. Log saved.")

