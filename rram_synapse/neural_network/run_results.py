
import numpy as np
import os
import copy
import threading
import argparse

from results import get_runs

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--print', type=int, default=0)
cmd_args = parser.parse_args()

##############################################

def run_command(param):
    
    name = '%s_%f_%f_%d_%d_%d_%f_%f' % (param['benchmark'], param['dt'], param['dt_scale'], param['step'], param['hidden'], param['dfa'], param['vc_scale'], param['rate'])
    
    cmd = "python %s --epochs %d --dt %f --dt_scale %f --step %d --hidden %d --dfa %d --vc_scale %f --rate %f --name %s" % \
        (param['benchmark'], param['epochs'], param['dt'], param['dt_scale'], param['step'], param['hidden'], param['dfa'], param['vc_scale'], param['rate'], name)

    if cmd_args.print:
        print (cmd)
    else:
        os.system(cmd)

    return

##############################################

runs = get_runs()

##############################################

num_runs = len(runs)
parallel_runs = 1

for run in range(0, num_runs, parallel_runs):
    threads = []
    for parallel_run in range( min(parallel_runs, num_runs - run)):
        args = runs[run + parallel_run]
        t = threading.Thread(target=run_command, args=(args,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        
