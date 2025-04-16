import subprocess
import os

rat_path = '/home/claramariadima/SNO/rat'
sudo_pass = 'BiPo214intheneck'

# === Create the shell script to run inside the container ===
container_script_path = os.path.join(rat_path, 'run_rat_macro.sh')
with open(container_script_path, 'w') as f:
    f.write("#!/bin/sh\n")
    f.write("cd /rat\n")
    f.write("scons -j8\n")
    f.write("rat get_pmt_positions.mac\n")

os.chmod(container_script_path, 0o755)

# === Build the Docker command ===
docker_cmd = (
    f"sudo -S docker run --init --rm "
    f"-v {rat_path}:/rat "
    f"-w /rat "
    f"snoplus/rat-container:root6 "
    f"./run_rat_macro.sh"
)

print("\n--- Running scons + RAT macro in Docker container ---")

# === Use Popen to see live output ===
process = subprocess.Popen(
    docker_cmd,
    shell=True,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Pipe the sudo password
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

