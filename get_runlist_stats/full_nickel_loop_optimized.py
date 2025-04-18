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

# === Setup logging ===

def random_log_name():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"log_{timestamp}_{rand}.txt"

log_filename = random_log_name()
log_file = open(log_filename, "w")

class Logger(object):
    def __init__(self, stream, logfile):
        self.terminal = stream
        self.logfile = logfile
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

sys.stdout = Logger(sys.stdout, log_file)
sys.stderr = Logger(sys.stderr, log_file)

print('Setting paths, filenames and commands')

# --- Paths ---
grid_singularity_path = '/home/claramariadima/SNO/Singularititty'
grid_tools_path       = '/home/claramariadima/SNO/rat-tools/GridTools'
grid_downloaded_path  = '/home/claramariadima/SNO/rat-tools/GridTools/downloaded'
phi_theta_path        = '/home/claramariadima/SNO/RS_isotropy/get_runlist_stats/phi_theta'
get_runlist_stats_path= '/home/claramariadima/SNO/RS_isotropy/get_runlist_stats'
rat_path              = '/home/claramariadima/SNO/rat'

# The singularity image is located in grid_singularity_path.
snopl_image           = 'snopl7.img'
# Build the full path to the singularity image:
singularity_image_path = os.path.join(grid_singularity_path, snopl_image)

# --- Credentials ---
couch_user = 'snoplus'
couch_pass = 'BiPo214intheneck'      # <-- Replace with your real password
sudo_pass  = ':18kyorosteaua18:'  # <-- Replace with your real sudo password

# --- Runlist ---
# You may use a test runlist or load from a file.
# Example: 
# test_runlist = [365063, 365066]
runlist = np.loadtxt('nickel_runlist_remaining.txt', dtype=int)
# runlist = test_runlist

def run_cmd(cmd, shell=True, cwd=None):
    print(f"\n--- Running command: {cmd} ---")
    result = subprocess.run(cmd, shell=shell, cwd=cwd)
    if result.returncode != 0:
        print(f"!!! Command failed: {cmd}")
    print(f"--- Finished command: {cmd} ---\n")
    return result.returncode

###############################################################################
# Function to download one run's zdab using a separate filelist.
###############################################################################
def download_run(run):
    """
    Download zdab file for a specific run.
    Uses raw_list with the -o option to write to filelist_{run}.dat in grid_tools_path,
    modifies that file, and then runs grabber with it.
    """
    
    # Randomized delay to stagger the raw_list calls.
    delay = random.uniform(1, 6)  # wait between 1 and 6 seconds
    print(f"Run {run}: Sleeping for {delay:.1f} seconds before starting raw_list.")
    time.sleep(delay)
    
    print(f"\n=== Starting download for run {run} ===")
    
    # Define a unique filelist name for this run.
    temp_filelist = f"filelist_{run}.dat"
    expect_script_path = os.path.join(grid_singularity_path, f'run_expect_{run}.exp')
    # Use the -o option to generate filelist_{run}.dat in grid_tools_path.
    raw_list_cmd = f"python3 raw_list -type L1 -r {run} {run+1} -o {temp_filelist}"
    
    # Build the singularity command.
    singularity_raw_list_cmd = (
        f"singularity exec -B /run/shm {singularity_image_path} /bin/bash -c "
        f"\"export MYPROXY_SERVER=myproxy.gridpp.rl.ac.uk && "
        f"export LCG_GFAL_INFOSYS=topbdii.grid.help.ph.ic.ac.uk:2170,lcgbdii.gridpp.rl.ac.uk:2170 && "
        f"cd {grid_tools_path} && {raw_list_cmd}\""
    )
    
    # Write the raw_list expect script for this run.
    with open(expect_script_path, 'w') as f:
        f.write(f"""#!/usr/bin/expect -f
set timeout -1
spawn {singularity_raw_list_cmd}
expect "Username:"
send "{couch_user}\r"
expect "Password:"
send "{couch_pass}\r"
expect eof
""")
    os.chmod(expect_script_path, 0o755)
    
    # Run the raw_list expect script.
    run_cmd(f"expect {expect_script_path}", cwd=grid_tools_path)
    
    # === Modify the temporary filelist (filelist_{run}.dat) ===
    temp_filelist_path = os.path.join(grid_tools_path, temp_filelist)
    with open(temp_filelist_path, "r") as f:
        lines = f.readlines()
    if lines:
        first_line = lines[0]
        parts = first_line.strip().split("\t")
        if len(parts) >= 3:
            old_url = parts[2]
            modified_url = re.sub(r".*/l1", "srm://lcg-snopse1.sfu.computecanada.ca:8443/snoplus/snotflow/raw/l1", old_url)
            parts[2] = modified_url
            modified_line = "\t".join(parts)
            with open(temp_filelist_path, "w") as f:
                f.write(modified_line + "\n")
            print(f"filelist_{run}.dat updated:\n{modified_line}")
        else:
            print(f"⚠️ First line in filelist_{run}.dat doesn't have expected format.")
    else:
        print(f"⚠️ filelist_{run}.dat is empty!")
    
    # === Run grabber with retry on timeout error ===
    expect_grabber_script_path = os.path.join(grid_singularity_path, f'run_grabber_{run}.exp')
    grabber_cmd = f"python3 grabber -l {temp_filelist} --streams 1 --timeout 144000"
    singularity_grabber_cmd = (
        f"singularity exec -B /run/shm {singularity_image_path} /bin/bash -c "
        f"\"export MYPROXY_SERVER=myproxy.gridpp.rl.ac.uk && "
        f"export LCG_GFAL_INFOSYS=topbdii.grid.help.ph.ic.ac.uk:2170,lcgbdii.gridpp.rl.ac.uk:2170 && "
        f"cd {grid_tools_path} && {grabber_cmd}\""
    )
    with open(expect_grabber_script_path, 'w') as f:
        f.write(f"""#!/usr/bin/expect -f
set timeout -1
spawn {singularity_grabber_cmd}
expect "download"
send "y\r"
expect eof
""")
    os.chmod(expect_grabber_script_path, 0o755)
    
    # Retry loop for grabber
    retry_count = 0
    while True:
        print(f"Running grabber (attempt {retry_count+1}) for run {run}")
        result = subprocess.run(f"expect {expect_grabber_script_path}", shell=True, cwd=grid_tools_path, capture_output=True, text=True)
        output = result.stdout + result.stderr
        if 'gfal-copy error: 110 (Connection timed out) - Transfer canceled because the timeout expired' in output:
            retry_count += 1
            print(f"Timeout error detected on grabber for run {run}, retrying... (attempt {retry_count})")
            time.sleep(5)
            continue
        if result.returncode != 0:
            raise RuntimeError(f"grabber step failed for run {run}: {output}")
        break

    # 2) Now wait for the zdab file to really be there and finished
    zdab_filename = f"SNOP_0000{run}_000.zdab"
    zdab_path = os.path.join(grid_downloaded_path, zdab_filename)

    print(f"Waiting for {zdab_filename} to appear…")
    start = time.time()
    timeout = 14400  # e.g. 1 hour max
    while True:
        if os.path.exists(zdab_path):
            size1 = os.path.getsize(zdab_path)
            if size1 > 0:
                # wait a few seconds and check it hasn’t grown
                time.sleep(5)
                size2 = os.path.getsize(zdab_path)
                if size2 == size1:
                    print(f"{zdab_filename} is fully downloaded ({size2} bytes).")
                    break
        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out waiting for {zdab_filename}")
        time.sleep(5)

    # 3) cleanup temp expect scripts
    try:
        os.remove(expect_script_path)
        os.remove(expect_grabber_script_path)
    except OSError:
        pass
    
    # 4) cleanup the temp filelist
    try:
        os.remove(temp_filelist_path)
        print(f"Deleted temporary filelist: {temp_filelist}")
    except OSError as e:
        print(f"⚠️ Could not delete {temp_filelist}: {e}")

    print(f"=== Download completed for run {run} ===")
    return run

###############################################################################
# Function to process one run's downloaded zdab and run RAT processing.
###############################################################################
def process_run(run):
    """
    Process a single run:
     - Moves the downloaded zdab file to the RAT directory.
     - Creates a temporary macro file for the run.
     - Builds and runs RAT in Docker.
     - Cleans up temporary files and organizes outputs.
    """
    print(f"\n=== Starting RAT processing for run {run} ===")
    
    # --- Ensure the RAT repo is on the correct branch ---
    desired_branch = "get_pmt_plots"
    result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                              cwd=rat_path, capture_output=True, text=True)
    current_branch = result.stdout.strip()
    if current_branch != desired_branch:
        print(f"Switching RAT repo from branch '{current_branch}' to '{desired_branch}' for run {run}")
        subprocess.run(["git", "checkout", desired_branch], cwd=rat_path)
    else:
        print(f"Already on branch: {desired_branch} for run {run}")
    
    # --- Move the downloaded zdab file to the RAT directory ---
    zdab_filename = f"SNOP_0000{run}_000.zdab"
    src_path = os.path.join(grid_downloaded_path, zdab_filename)
    dst_path = os.path.join(rat_path, zdab_filename)
    if os.path.exists(src_path):
        os.rename(src_path, dst_path)
        print(f"Moved {zdab_filename} to RAT directory for run {run}.")
        # Delete the corresponding filelist in GridTools
        filelist_to_delete = os.path.join(grid_tools_path, f"filelist_{run}.dat")
        if os.path.exists(filelist_to_delete):
            try:
                os.remove(filelist_to_delete)
                print(f"Deleted filelist_{run}.dat from GridTools folder for run {run}.")
            except Exception as e:
                print(f"⚠️ Could not delete filelist_{run}.dat: {e}")
    else:
        print(f"⚠️ zdab file not found at: {src_path} for run {run}")
    
    # --- Create a temporary macro file for this run ---
    macro_template_path = os.path.join(rat_path, "get_pmt_positions.mac")
    temp_macro_path = os.path.join(rat_path, f"get_pmt_positions_{run}.mac")
    updated_lines = []
    with open(macro_template_path, "r") as f:
        for line in f:
            if line.startswith("/rat/inzdab/load"):
                updated_lines.append(f"/rat/inzdab/load {zdab_filename}\n")
            else:
                updated_lines.append(line)
    with open(temp_macro_path, "w") as f:
        f.writelines(updated_lines)
    print(f"Created temporary macro file {temp_macro_path} for run {run}.")
    
    # --- Build and run RAT inside Docker ---
    print(f"\n--- Building RAT and running macro in Docker for run {run} ---")
    container_script_path = os.path.join(rat_path, f'run_rat_macro_{run}.sh')
    with open(container_script_path, 'w') as f:
        f.write("#!/bin/sh\n")
        f.write("cd /rat\n")
        f.write("scons -j8\n")
        # Use the temporary macro file for this run.
        f.write(f"rat get_pmt_positions_{run}.mac\n")
    os.chmod(container_script_path, 0o755)
    
    docker_cmd = (
        f"sudo -S docker run --init --rm "
        f"-v {rat_path}:/rat "
        f"-w /rat "
        f"snoplus/rat-container:root6 "
        f"./run_rat_macro_{run}.sh"
    )
    process = subprocess.Popen(
        docker_cmd,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    process.stdin.write(sudo_pass + "\n")
    process.stdin.flush()
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"\n❌ Docker container exited with error code: {process.returncode} for run {run}")
    else:
        print(f"\n✅ Docker container completed successfully for run {run}!")
    
    # --- Clean up temporary macro and container script ---
    try:
        os.remove(temp_macro_path)
        os.remove(container_script_path)
        print(f"Temporary macro and container script deleted for run {run}.")
    except Exception as e:
        print(f"⚠️ Error deleting temporary files for run {run}: {e}")
    
    # --- Organize output files and clean up ---
    print(f"\n--- Cleaning up outputs for run {run} ---")
    # 1. Delete the zdab file from the RAT directory.
    if os.path.exists(dst_path):
        os.remove(dst_path)
        print(f"Deleted {zdab_filename} from RAT directory for run {run}.")
    else:
        print(f"⚠️ Expected zdab file not found at: {dst_path} for run {run}")
    
    # 2. Move {run}_phi_theta.csv to the phi_theta folder.
    phi_theta_filename = f"{run}_phi_theta.csv"
    phi_theta_src = os.path.join(rat_path, phi_theta_filename)
    phi_theta_dst = os.path.join(phi_theta_path, phi_theta_filename)
    if os.path.exists(phi_theta_src):
        os.rename(phi_theta_src, phi_theta_dst)
        print(f"Moved {phi_theta_filename} to phi_theta folder for run {run}.")
    else:
        print(f"⚠️ Expected phi_theta file not found at: {phi_theta_src} for run {run}")
    
    # 3. Delete {run}_indices.txt.
    indices_filename = f"{run}_indices.txt"
    indices_path = os.path.join(rat_path, indices_filename)
    if os.path.exists(indices_path):
        os.remove(indices_path)
        print(f"Deleted {indices_filename} from RAT directory for run {run}.")
    else:
        print(f"⚠️ Expected indices file not found at: {indices_path} for run {run}")
    
    # 4. Delete all rat.*.log files.
    log_files = glob.glob(os.path.join(rat_path, "rat.*.log"))
    if log_files:
        for lfile in log_files:
            os.remove(lfile)
            print(f"Deleted log file {os.path.basename(lfile)} for run {run}.")
    else:
        print(f"No rat.*.log files found for run {run}.")
    
    # 5. Delete DQHL tables (.root and .ratdb).
    for ext in ["root", "ratdb"]:
        dqhl_table_filename = f"DATAQUALITY_RECORDS_{run}_p1.{ext}"
        dqhl_table_path = os.path.join(rat_path, dqhl_table_filename)
        if os.path.exists(dqhl_table_path):
            os.remove(dqhl_table_path)
            print(f"Deleted {dqhl_table_filename} for run {run}.")
        else:
            print(f"⚠️ Expected {dqhl_table_filename} not found for run {run}.")
    
    # 6. Delete applied_plot and flags_plot files.
    for tag in ["applied", "flags"]:
        plot_filename = f"{tag}_{run}_p1.png"
        plot_path = os.path.join(rat_path, plot_filename)
        if os.path.exists(plot_path):
            os.remove(plot_path)
            print(f"Deleted {plot_filename} for run {run}.")
        else:
            print(f"⚠️ Expected {plot_filename} not found for run {run}.")
    
    # 7. Delete PMT coverage maps.
    for name in ['PMT coverage map before PMT Calibrations.png', 'PMT coverage map after PMT Calibrations.png']:
        path_ = os.path.join(rat_path, name)
        if os.path.exists(path_):
            os.remove(path_)
            print(f"Deleted {name} for run {run}.")
        else:
            print(f"⚠️ Expected {name} not found for run {run}.")
    
    # 8. Delete GeoCoverageMaps.
    for fname in [f"TGraph_GeoCoverageMap.png", f"TGraph_GeoCoverageMap_{run}_p1.png"]:
        path_ = os.path.join(rat_path, fname)
        if os.path.exists(path_):
            os.remove(path_)
            print(f"Deleted {fname} for run {run}.")
        else:
            print(f"⚠️ Expected {fname} not found for run {run}.")
    
    # 9. Delete Crate Coverage maps.
    for fname in [f"TH2D_CrateCoverageMap_{run}_p1.png", f"TH2D_UnCalCrateCoverageMap_{run}_p1.png"]:
        path_ = os.path.join(rat_path, fname)
        if os.path.exists(path_):
            os.remove(path_)
            print(f"Deleted {fname} for run {run}.")
        else:
            print(f"⚠️ Expected {fname} not found for run {run}.")
    
    # 10. Final step: run compute_run_metrics.py.
    compute_stats_cmd = f"python3 compute_run_metrics.py -run_number {run}"
    run_cmd(compute_stats_cmd, cwd=get_runlist_stats_path)
    
    print(f"=== RAT processing complete for run {run} ===\n")
    return run

###############################################################################
# Main flow: Download and process runs as they complete using as_completed.
###############################################################################
if __name__ == "__main__":
    print("\n--- Starting parallel download and pipelined processing ---\n")

    # How many downloads can run at once?
    MAX_DOWNLOADS = 6
    # How many Docker/process jobs can run at once?
    MAX_PROCS     = 6

    # Pool #1: handles only downloads
    download_executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_DOWNLOADS)
    # Pool #2: handles only processing
    process_executor  = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PROCS)

    # Kick off ALL the download tasks
    future_to_run = {
        download_executor.submit(download_run, run): run
        for run in runlist
    }

    # As soon as each download finishes, immediately queue its processing
    for dl_future in concurrent.futures.as_completed(future_to_run):
        run_number = dl_future.result()   # will raise if download failed
        print(f"\n>> Download finished for run {run_number}; queueing processing...\n")
        process_executor.submit(process_run, run_number)

    # Once we've submitted every download, wait for them all to truly finish
    download_executor.shutdown(wait=True)
    # Then wait for every processing job to finish
    process_executor.shutdown(wait=True)

    log_file.close()
    print(f"\nAll done! Output was saved to: {log_filename}")
