
import numpy as np
import os
import copy
import threading
import argparse

################################################

def get_perms(param):
    params = [param]
    
    for key in param.keys():
        val = param[key]
        if type(val) == list:
            new_params = []
            for ii in range(len(val)):
                for jj in range(len(params)):
                    new_param = copy.copy(params[jj])
                    new_param[key] = val[ii]
                    new_params.append(new_param)
                    
            params = new_params
            
    return params

################################################

network = {'benchmark':'network.py', 'epochs':10, 'dt':[1e-4, 3e-4, 1e-5, 3e-5], 'dt_scale':[1.], 'step':1, 'hidden':[100, 300, 500], 'dfa':[1, 0], 'vc_scale':1., 'rate':[0.0]}

###############################################

params = [network]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
