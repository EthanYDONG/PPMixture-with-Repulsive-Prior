import numpy as np
import os
# print('CWDDDDD:',os.getcwd())
# import sys
# print('SYSSSS:',sys.path)
from .Simulation_Thinning_Poisson import Simulation_Thinning_Poisson
from .ImpactFunction import ImpactFunction 


def Simulation_Branch_HP(para, options):
    """
    Simulate Hawkes processes as Branch processes.

    Reference:
    Møller, Jesper, and Jakob G. Rasmussen.
    "Approximate simulation of Hawkes processes."
    Methodology and Computing in Applied Probability 8.1 (2006): 53-64.

    """
    Seqs_list = []
    
    for n in range(1, options['N'] + 1):
        Seqs = {'Time': [], 'Mark': [], 'Start': [], 'Stop': [], 'Feature': []}
        # the 0-th generation, simulate exogenous events via Poisson processes
        History = Simulation_Thinning_Poisson(para['mu'], 0, options['Tmax'])
        History = np.squeeze(History)
        current_set = History.copy()
        
        for k in range(1, options['GenerationNum'] + 1):
            future_set = []
            for i in range(current_set.shape[1]):
                ti = current_set[0, i]
                ui = int(current_set[1, i])
                
                t = 0
                phi_t = ImpactFunction(ui, t, para)
                mt = np.sum(phi_t)
               
                while t < options['Tmax'] - ti:
                    s = np.random.exponential(1 / mt)
                    U = np.random.rand()
                    phi_ts = ImpactFunction(ui, [t + s], para)
                    mts = np.sum(phi_ts)

                    if t + s > options['Tmax'] - ti or U > mts / mt:
                        t = t + s
                    else:
                        u = np.random.rand() * mts
                        sumIs = 0
                        for d in range(len(phi_ts)):
                            sumIs = sumIs + phi_ts[d]
                            if sumIs >= u:
                                break
                        index = d
                        

                        t = t + s
                        #import pdb;pdb.set_trace()

                        future_set.append([t + ti, index])
                    #import pdb;pdb.set_trace()
                    t = t
                    phi_t = ImpactFunction(ui, t, para)
                    mt = np.sum(phi_t)

            if len(future_set) == 0 or History.shape[1] > options['Nmax']:
                break
            else:
                current_set = future_set.copy()
                transposed_current_set = np.array(current_set).T
                History = np.hstack([History, transposed_current_set])
                current_set = transposed_current_set

        index = np.argsort(History[0, :])
        Seqs['Time'] = History[0, index]
        Seqs['Mark'] = History[1, index]
        Seqs['Start'] = 0
        Seqs['Stop'] = options['Tmax']
        index = np.where(Seqs['Time'] <= options['Tmax'])
        Seqs['Time'] = Seqs['Time'][index]
        Seqs['Mark'] = Seqs['Mark'][index]
        Seqs_list.append(Seqs)

        if n % 10 == 0 or n == options['N']:
            print(f"#seq={n}/{options['N']}, #event={len(Seqs['Mark'])}")

    return Seqs_list
