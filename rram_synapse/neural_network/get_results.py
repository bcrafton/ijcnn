
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
    name = '%s_%f_%f_%d_%d_%d_%f.npy' % (param['benchmark'], param['dt'], param['dt_scale'], param['step'], param['hidden'], param['dfa'], param['vc_scale'])
    
    # load the results
    try:
        res = np.load(name).item()
        
        key = (param['benchmark'], param['dfa'], param['hidden'])
        res = res['train_acc']
        res = np.reshape(res, (-1, 60))
        res = res[:, -1]
        
        val = max(res)
        print (res)
        print (name, val)
        
        if key in results.keys():
            # use an if instead of max because we gonna want to save the winner run information
            if results[key][0] < val:
                results[key] = (val, name)
        else:
            results[key] = (val, name)
    except:
        pass
            
for key in sorted(results.keys()):   
    print (key, results[key])





        
