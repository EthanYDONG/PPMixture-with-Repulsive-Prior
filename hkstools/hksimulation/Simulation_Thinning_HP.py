import numpy as np
from .SupIntensity_HP import SupIntensity_HP
from .Intensity_HP import Intensity_HP
import time 

def Simulation_Thinning_HP(para, options):
    Seqs = {'Time': [], 'Mark': [], 'Start': [], 'Stop': [], 'Feature': []}
    tic = time.time()
    for n in range(1, options['N']+1):
        t = 0
        History = np.array([]).reshape(2, 0)

        mt = SupIntensity_HP(t, History, para, options)

        while t < options['Tmax'] and History.shape[1] < options['Nmax']:
            s = np.random.exponential(1/mt)
            U = np.random.rand()

            lambda_ts = Intensity_HP(t + s, History, para)
            mts = np.sum(lambda_ts)

            if t + s > options['Tmax'] or U > mts/mt:
                t = t + s
            else:
                u = np.random.rand() * mts
                sumIs = 0

                for d in range(len(lambda_ts)):
                    sumIs = sumIs + lambda_ts[d]
                    if sumIs >= u:
                        break
                index = d

                t = t + s
                History = np.column_stack((History, np.array([t, index+1])))

        Seqs['Time'].append(History[0, :].tolist())
        Seqs['Mark'].append(History[1, :].tolist())
        Seqs['Start'].append(0)
        Seqs['Stop'].append(options['Tmax'])

        index = np.where(np.array(Seqs['Time'][n-1]) <= options['Tmax'])[0]
        Seqs['Time'][n-1] = [Seqs['Time'][n-1][i] for i in index]
        Seqs['Mark'][n-1] = [Seqs['Mark'][n-1][i] for i in index]

        if n % 10 == 0 or n == options['N']:
            print('#seq={}/{}, #event={}, time={:.2f}sec'.format(
                n, options['N'], len(Seqs['Time'][n-1]), time.time()-tic))

    return Seqs