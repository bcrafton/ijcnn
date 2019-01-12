
import numpy as np
import os
import threading

def run_command(epoch, lr):
    cmd = "python mnist_inh_synapses.py --epochs %d --lr %0.12f" % (epoch, lr)
    os.system(cmd)
    return

################################################

epochs = [10]

lrs = np.linspace(-10., 0., 12)
lrs = np.power(10., lrs)

runs = []
for epoch in epochs:
    for lr in lrs:
        runs.append((epoch, lr))

num_runs = len(runs)
parallel_runs = 12

for run in range(0, num_runs, parallel_runs):
    threads = []
    for parallel_run in range(parallel_runs):
        args = runs[run + parallel_run]
        t = threading.Thread(target=run_command, args=args)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

################################################



