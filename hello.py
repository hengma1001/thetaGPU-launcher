import socket
import subprocess
import os

print("Hostname:", socket.gethostname())
print("CUDA Envs:", {k:v for k,v in os.environ.items() if "CUDA" in k})
subprocess.run(["nvidia-smi", "-L"])
