import numpy as np
import cudf

# Both import methods supported
from cuml import LinearRegression
from cuml.linear_model import LinearRegression
import socket
import subprocess
import os
import atexit



DIM = 9_000_000

print("Hostname:", socket.gethostname())
print("PID:", os.getpid())
GPU_ID = os.environ["CUDA_VISIBLE_DEVICES"]
print("CUDA_VISIBLE_DEVICES is set to:", GPU_ID)
#proc = subprocess.Popen("nvidia-smi -i {GPU_ID} -l 1", shell=True)
proc = subprocess.Popen(["nvidia-smi", "-l", "5", "-i", GPU_ID])

def onexit():
    proc.terminate()
    proc.kill()

atexit.register(onexit)

lr = LinearRegression(fit_intercept = True, normalize = False, algorithm = "eig")

X = cudf.DataFrame()
X['col1'] = np.random.rand(DIM)
X['col2'] = np.random.rand(DIM)

y = cudf.Series(np.random.rand(DIM)) 

reg = lr.fit(X,y)

X_new = cudf.DataFrame()
X_new['col1'] = np.random.rand(DIM)
X_new['col2'] = np.random.rand(DIM)
preds = lr.predict(X_new)
print("DONE!")

