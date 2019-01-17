
import numpy as np
import os
import copy
import threading
import argparse
from results import get_runs

##############################################

runs = get_runs()

##############################################

results = {}

num_runs = len(runs)
for ii in range(num_runs):
    param = runs[ii]

    # figure out the name of the param
    name = '%s_%f_%f_%d_%d_%d_%f' % (param['benchmark'], param['dt'], param['dt_scale'], param['step'], param['hidden'], param['dfa'], param['vc_scale'])

    # load the results
    res = np.load(name).item()
    
    key = (param['benchmark'], param['dfa'], param['hidden'], transfer)
    val = max(res['test_acc'])

    print (name, val)
    
    if key in results.keys():
        # use an if instead of max because we gonna want to save the winner run information
        if results[key][0] < val:
            results[key] = (val, name)
    else:
        results[key] = (val, name)
            
for key in sorted(results.keys()):   
    print (key, results[key])





        
