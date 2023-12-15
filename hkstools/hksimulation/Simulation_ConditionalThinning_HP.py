import numpy as np
from .SupIntensity_HP import SupIntensity_HP
from .Intensity_HP import Intensity_HP
import time

def Simulation_ConditionalThinning_HP(SeqsOld, para, options):
    """
    Implementation of Ogata's thinning method to simulate Hawkes processes
    conditioned on Historical event sequences (SeqsOld)

    Reference:
    Ogata, Yosihiko. "On Lewis' simulation method for point processes." 
    IEEE Transactions on Information Theory 27.1 (1981): 23-31.

    Provider:
    Hongteng Xu @ Georgia Tech
    June 10, 2017
    """

    SeqsNew = {'Time': [],
               'Mark': [],
               'Start': [],
               'Stop': [],
               'Feature': []}
    tic = time.time()
    for n in range(len(SeqsOld)):
        t = SeqsOld[n]['Stop']
        History = np.vstack((SeqsOld[n]['Time'], SeqsOld[n]['Mark']))

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

            mt = SupIntensity_HP(t, History, para, options)

        SeqsNew[n] = {'Time': History[0, :].tolist(),
                      'Mark': History[1, :].tolist(),
                      'Start': SeqsOld[n]['Stop'],
                      'Stop': options['Tmax']}

        index = np.where((SeqsOld[n]['Stop'] <= SeqsNew[n]['Time']) & (SeqsNew[n]['Time'] <= options['Tmax']))
        SeqsNew[n]['Time'] = np.array(SeqsNew[n]['Time'])[index].tolist()
        SeqsNew[n]['Mark'] = np.array(SeqsNew[n]['Mark'])[index].tolist()

        if n % 10 == 0 or n == options['N']:
            print(f'#seq={n}/{options["N"]}, #event={len(SeqsNew[n]["Mark"])}, time={time.time()-tic:.2f}sec')

    return SeqsNew
