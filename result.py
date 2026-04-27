import subprocess, sys, os

# Use the .venv python which has CUDA
venv_python = r'C:\acm\AdoDAS2026-main\.venv\Scripts\python.exe'
train_script = r'C:\acm\AdoDAS2026-main\train.py'
config = r'C:\acm\AdoDAS2026-main\tasks\a1\sch002_sch003_quick.yaml'

log_path = r'C:\acm\AdoDAS2026-main\train_run_log.txt'

# Change to project directory
os.chdir(r'C:\acm\AdoDAS2026-main')

cmd = [venv_python, train_script, '--task', 'a1', '--config', config]

with open(log_path, 'w', encoding='utf-8') as log:
    log.write(f"Running: {' '.join(cmd)}\n")
    log.write(f"CWD: {os.getcwd()}\n\n")
    log.flush()
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    for line in process.stdout:
        log.write(line)
        log.flush()
    
    process.wait()
    log.write(f"\nReturn code: {process.returncode}\n")