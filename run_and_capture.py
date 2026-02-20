import os
import subprocess

# Set environment variables
env = os.environ.copy()
env["OPENBLAS_NUM_THREADS"] = "1"
env["OMP_NUM_THREADS"] = "4"

print("Running verification and capturing output to result.txt...")
with open("result.txt", "w", encoding="utf-8") as f:
    process = subprocess.Popen(
        ["python", "verify_arrow_decoder.py"],
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True
    )
    process.wait()

print("Done. Check result.txt")
