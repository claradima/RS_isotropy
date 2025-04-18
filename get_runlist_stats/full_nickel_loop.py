import os
import sys
import subprocess
import numpy as np
import datetime
import random
import string
import re
import glob

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

# === Script starts here ===
print('Setting paths, filenames and commands')

# --- Paths ---
grid_singularity_path = '/home/claramariadima/SNO/Singularititty'
grid_tools_path = '/home/claramariadima/SNO/rat-tools/GridTools'
grid_downloaded_path = '/home/claramariadima/SNO/rat-tools/GridTools/downloaded'
phi_theta_path = '/home/claramariadima/SNO/RS_isotropy/get_runlist_stats/phi_theta'
get_runlist_stats_path = '/home/claramariadima/SNO/RS_isotropy/get_runlist_stats'
rat_path = '/home/claramariadima/SNO/rat'

snopl_image = 'snopl7.img'

# --- Credentials ---
couch_user = 'snoplus'
couch_pass = 'BiPo214intheneck'  # <-- Replace with real password
sudo_pass = ':18kyorosteaua18:'  # <-- Replace with real password

# --- Runlist ---
test_runlist = [365063, 365066]
runlist = np.loadtxt('nickel_runlist.txt', dtype=int)
#runlist = test_runlist

def run_cmd(cmd, shell=True, cwd=None):
    print(f"\n--- Running command: {cmd} ---")
    result = subprocess.run(cmd, shell=shell, cwd=cwd)
    if result.returncode != 0:
        print(f"!!! Command failed: {cmd}")
    print(f"--- Finished command: {cmd} ---\n")
    return result.returncode

for run in runlist:
    print(f'Starting process of downloading zdab for run {run}')

    expect_script_path = os.path.join(grid_singularity_path, 'run_expect.exp')
    raw_list_cmd = f"python3 raw_list -type L1 -r {run} {run + 1}"
    
    # Build the full singularity command to run directly
    singularity_raw_list_cmd = (
        f"singularity exec -B /run/shm {snopl_image} /bin/bash -c "
        f"\"export MYPROXY_SERVER=myproxy.gridpp.rl.ac.uk && "
        f"export LCG_GFAL_INFOSYS=topbdii.grid.help.ph.ic.ac.uk:2170,lcgbdii.gridpp.rl.ac.uk:2170 && "
        f"cd {grid_tools_path} && "
        f"{raw_list_cmd}\""
    )

    # Write the Expect script
    with open(expect_script_path, 'w') as f:
        f.write(f"""#!/usr/bin/expect -f
set timeout -1
spawn {singularity_raw_list_cmd}
expect "overwrite"
send "y\\r"
expect "Username:"
send "{couch_user}\\r"
expect "Password:"
send "{couch_pass}\\r"
expect eof
""")

    os.chmod(expect_script_path, 0o755)

    # Run the expect script on the host
    run_cmd(f"expect {expect_script_path}", cwd=grid_singularity_path)
    
    # === Modify filelist.dat ===
    filelist_path = os.path.join(grid_tools_path, "filelist.dat")

    with open(filelist_path, "r") as f:
        lines = f.readlines()

    if lines:
        first_line = lines[0]
    
        # Replace any prefix ending with /l1 with the new srm path
        parts = first_line.strip().split("\t")
        if len(parts) >= 3:
            old_url = parts[2]
            modified_url = re.sub(r".*/l1", "srm://lcg-snopse1.sfu.computecanada.ca:8443/snoplus/snotflow/raw/l1", old_url)
            parts[2] = modified_url
            modified_line = "\t".join(parts)
        
            with open(filelist_path, "w") as f:
                f.write(modified_line + "\n")

            print(f"filelist.dat updated: kept 1 line, changed URL to:\n{modified_line}")
        else:
            print("⚠️ First line in filelist.dat doesn't have expected format.")
    else:
        print("⚠️ filelist.dat is empty!")
        
    # === Run grabber inside container with expect ===
    expect_grabber_script_path = os.path.join(grid_singularity_path, 'run_grabber.exp')
    grabber_cmd = "python3 grabber -l filelist.dat"

    singularity_grabber_cmd = (
    f"singularity exec -B /run/shm {snopl_image} /bin/bash -c "
    f"\"export MYPROXY_SERVER=myproxy.gridpp.rl.ac.uk && "
    f"export LCG_GFAL_INFOSYS=topbdii.grid.help.ph.ic.ac.uk:2170,lcgbdii.gridpp.rl.ac.uk:2170 && "
    f"cd {grid_tools_path} && "
    f"{grabber_cmd}\""
    )

    with open(expect_grabber_script_path, 'w') as f:
        f.write(f"""#!/usr/bin/expect -f
    set timeout -1
    spawn {singularity_grabber_cmd}
    expect "download"
    send "y\\r"
    expect eof
    """)

    os.chmod(expect_grabber_script_path, 0o755)
    run_cmd(f"expect {expect_grabber_script_path}", cwd=grid_singularity_path)
    
    # === Ensure we're on the correct git branch in the rat repo ===
    desired_branch = "get_pmt_plots"
    
    # Check current branch
    result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=rat_path, capture_output=True, text=True)
    current_branch = result.stdout.strip()

    if current_branch != desired_branch:
        print(f"Switching rat repo from branch '{current_branch}' to '{desired_branch}'")
        subprocess.run(["git", "checkout", desired_branch], cwd=rat_path)
    else:
        print(f"Already on the correct git branch: {desired_branch}")

    # === Move downloaded zdab file to rat directory ===
    zdab_filename = f"SNOP_0000{run}_000.zdab"
    src_path = os.path.join(grid_downloaded_path, zdab_filename)
    dst_path = os.path.join(rat_path, zdab_filename)

    if os.path.exists(src_path):
        os.rename(src_path, dst_path)
        print(f"Moved {zdab_filename} to rat directory.")
    else:
        print(f"⚠️ zdab file not found at: {src_path}")
    
    
    # === Modify get_pmt_positions.mac with current run ===
    macro_path = os.path.join(rat_path, "get_pmt_positions.mac")
    updated_lines = []

    with open(macro_path, "r") as f:
        for line in f:
            if line.startswith("/rat/inzdab/load"):
                updated_line = f"/rat/inzdab/load SNOP_0000{run}_000.zdab\n"
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)

    with open(macro_path, "w") as f:
        f.writelines(updated_lines)

    print(f"Updated get_pmt_positions.mac with run {run}")
    
    # === Build and run RAT inside Docker container ===
    print("\n--- Building RAT with scons and running macro in Docker container ---")

    # Create the shell script to run inside the container
    container_script_path = os.path.join(rat_path, 'run_rat_macro.sh')
    with open(container_script_path, 'w') as f:
        f.write("#!/bin/sh\n")
        f.write("cd /rat\n")
        f.write("scons -j8\n")
        f.write("rat get_pmt_positions.mac\n")

    os.chmod(container_script_path, 0o755)

    # Run the Docker container with the script
    docker_cmd = (
        f"sudo -S docker run --init --rm "
        f"-v {rat_path}:/rat "
        f"-w /rat "
        f"snoplus/rat-container:root6 "
        f"./run_rat_macro.sh"
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

    # Print live output
    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        print(f"\n❌ Docker container exited with error code: {process.returncode}")
    else:
        print("\n✅ Docker container completed successfully!")
    
    
    
    # === Clean up temporary script files ===
    try:
        os.remove(expect_script_path)
        os.remove(expect_grabber_script_path)
        os.remove(container_script_path)
        print("Temporary script files cleaned up.")
    except Exception as e:
        print(f"⚠️ Could not delete temporary files: {e}")
    
    
    
    # === Clean up and organize output files ===
    print("\n--- Cleaning up temporary files and moving useful outputs ---")

    # 1. Delete the .zdab file from rat folder
    if os.path.exists(dst_path):
        os.remove(dst_path)
        print(f"Deleted {zdab_filename} from rat directory.")
    else:
        print(f"⚠️ Expected zdab file not found at: {dst_path}")

    # 2. Move {run}_phi_theta.csv to phi_theta folder
    phi_theta_filename = f"{run}_phi_theta.csv"
    phi_theta_src = os.path.join(rat_path, phi_theta_filename)
    phi_theta_dst = os.path.join(phi_theta_path, phi_theta_filename)

    if os.path.exists(phi_theta_src):
        os.rename(phi_theta_src, phi_theta_dst)
        print(f"Moved {phi_theta_filename} to phi_theta folder.")
    else:
        print(f"⚠️ Expected phi_theta file not found at: {phi_theta_src}")   
        
    # 3. Delete {run}_indices.txt
    indices_filename = f"{run}_indices.txt"
    indices_path = os.path.join(rat_path, indices_filename)
    if os.path.exists(indices_path):
        os.remove(indices_path)
        print(f"Deleted {indices_filename} from rat directory.")
    else:
        print(f"⚠️ Expected indices file not found at: {indices_path}")
        
    
    # 4. Delete all rat.*.log files from rat folder
    log_files = glob.glob(os.path.join(rat_path, "rat.*.log"))
    if log_files:
        for log_file in log_files:
            os.remove(log_file)
            print(f"Deleted log file: {os.path.basename(log_file)}")
    else:
        print("No rat.*.log files found to delete.")
        
    # 5. Delete DQHL table
    dqhl_table_filename = f'DATAQUALITY_RECORDS_{run}_p1.root'
    dqhl_table_path = os.path.join(rat_path, dqhl_table_filename)
    if os.path.exists(dqhl_table_path):
        os.remove(dqhl_table_path)
        print(f"Deleted {dqhl_table_filename} from rat directory.")
    else:
        print(f"⚠️ Expected file not found at: {dqhl_table_path}")
        
    dqhl_table_filename = f'DATAQUALITY_RECORDS_{run}_p1.ratdb'
    dqhl_table_path = os.path.join(rat_path, dqhl_table_filename)
    if os.path.exists(dqhl_table_path):
        os.remove(dqhl_table_path)
        print(f"Deleted {dqhl_table_filename} from rat directory.")
    else:
        print(f"⚠️ Expected file not found at: {dqhl_table_path}")
        
    # 6. Delete applied_plot table
    applied_plot_filename = f'applied_{run}_p1.png'
    applied_plot_path = os.path.join(rat_path, applied_plot_filename)
    if os.path.exists(applied_plot_path):
        os.remove(applied_plot_path)
        print(f"Deleted {applied_plot_filename} from rat directory.")
    else:
        print(f"⚠️ Expected file not found at: {applied_plot_path}")
        
    # 7. Delete flags_plot table
    flags_plot_filename = f'flags_{run}_p1.png'
    flags_plot_path = os.path.join(rat_path, flags_plot_filename)
    if os.path.exists(flags_plot_path):
        os.remove(flags_plot_path)
        print(f"Deleted {flags_plot_filename} from rat directory.")
    else:
        print(f"⚠️ Expected file not found at: {flags_plot_path}")
        
    # 8. Delete PMT coverage maps
    map_before_calib = 'PMT coverage map before PMT Calibrations.png'
    map_after_calib = 'PMT coverage map after PMT Calibrations.png'
    map_before_calib_path = os.path.join(rat_path, map_before_calib)
    map_after_calib_path = os.path.join(rat_path, map_after_calib)
    
    if os.path.exists(map_before_calib_path):
        os.remove(map_before_calib_path)
        print(f"Deleted {map_before_calib} from rat directory.")
    else:
        print(f"⚠️ Expected file not found at: {map_before_calib_path}")
        
    if os.path.exists(map_after_calib_path):
        os.remove(map_after_calib_path)
        print(f"Deleted {map_after_calib} from rat directory.")
    else:
        print(f"⚠️ Expected file not found at: {map_after_calib_path}")
        
    # 9. Delete GeoCoverageMaps
    geomap_file = 'TGraph_GeoCoverageMap.png'
    geomap_run_file = f'TGraph_GeoCoverageMap_{run}_p1.png'
    geomap_path = os.path.join(rat_path, geomap_file)
    geomap_run_path = os.path.join(rat_path, geomap_run_file)
    
    if os.path.exists(geomap_path):
        os.remove(geomap_path)
        print(f"Deleted {geomap_path} from rat directory.")
    else:
        print(f"⚠️ Expected file not found at: {geomap_path}")
        
    if os.path.exists(geomap_run_path):
        os.remove(geomap_run_path)
        print(f"Deleted {geomap_run_path} from rat directory.")
    else:
        print(f"⚠️ Expected file not found at: {geomap_run_path}")
        
    # 10. Delete Crate Coverage maps
    cratecov = f'TH2D_CrateCoverageMap_{run}_p1.png'
    uncal_cratecov = f'TH2D_UnCalCrateCoverageMap_{run}_p1.png'
    cratecov_path = os.path.join(rat_path, cratecov)
    uncal_cratecov_path = os.path.join(rat_path, uncal_cratecov)
    
    if os.path.exists(cratecov_path):
        os.remove(cratecov_path)
        print(f"Deleted {cratecov_path} from rat directory.")
    else:
        print(f"⚠️ Expected file not found at: {cratecov_path}")
        
        
    if os.path.exists(uncal_cratecov_path):
        os.remove(uncal_cratecov_path)
        print(f"Deleted {uncal_cratecov_path} from rat directory.")
    else:
        print(f"⚠️ Expected file not found at: {uncal_cratecov_path}")
    
    
    # === Final step: run compute_run_metrics.py ===
    compute_stats_cmd = f"python3 compute_run_metrics.py -run_number {run}"
    run_cmd(compute_stats_cmd, cwd=get_runlist_stats_path)
    

log_filename.close()
print(f"\nAll done! Output was saved to: {log_filename}")


