
import numpy as np
import os
import threading

def run_command(epoch, dt, dt_scale, step):
    name = './results/dt_%0.10f_dt_scale_%f_step_%d' % (dt, dt_scale, step)
    cmd = "python network.py --epochs %d --dt %0.10f --dt_scale %f --step %d > %s" % (epoch, dt, dt_scale, step, name)
    os.system(cmd)
    return

################################################

epochs = [100]
# dts = [3e-4, 1e-4, 3e-5, 1e-5]
# dt_scales = [2., 4., 8.]

dts = [3e-5, 1e-5, 3e-6, 1e-6]
dt_scales = [4., 8., 10.]
steps = [1]

runs = []
for epoch in epochs:
    for dt in dts:
        for dt_scale in dt_scales:
            for step in steps:
                runs.append((epoch, dt, dt_scale, step))

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



