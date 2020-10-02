import socket
import subprocess
import os
import sys

print(sys.executable, sys.path)
print("Hostname:", socket.gethostname())
print("CUDA Envs:", {k:v for k,v in os.environ.items() if "CUDA" in k})
subprocess.run(["nvidia-smi", "-L"])
