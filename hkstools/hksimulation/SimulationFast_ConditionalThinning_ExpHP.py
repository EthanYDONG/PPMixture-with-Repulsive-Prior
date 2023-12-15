import numpy as np
from .Intensity_HP import Intensity_HP
from .Intensity_Recurrent_HP import Intensity_Recurrent_HP
import time 

def SimulationFast_ConditionalThinning_ExpHP(SeqsOld, para, options):
    SeqsNew = {'Time': [], 'Mark': [], 'Start': [], 'Stop': [], 'Feature': []}
    tic = time.time()
    for n in range(len(SeqsOld)):
        t = SeqsOld[n]['Stop']
        History = np.vstack((SeqsOld[n]['Time'], SeqsOld[n]['Mark']))

        lambdat = Intensity_HP(t, History, para)
        mt = np.sum(lambdat)

        while t < options['Tmax'] and History.shape[1] < options['Nmax']:
            s = np.random.exponential(1/mt)
            U = np.random.rand()

            lambda_ts = Intensity_Recurrent_HP(t + s, [], t, lambdat, para)
            mts = np.sum(lambda_ts)

            if t + s > options['Tmax'] or U > mts / mt:
                t = t + s
                lambdat = lambda_ts
            else:
                u = np.random.rand() * mts
                sumIs = 0
                for d in range(len(lambda_ts)):
                    sumIs = sumIs + lambda_ts[d]
                    if sumIs >= u:
                        break
                index = d

                lambdat = Intensity_Recurrent_HP(t + s, [index[0]], t, lambdat, para)
                t = t + s
                History = np.hstack((History, np.array([[t], [index[0]]])))

            mt = np.sum(lambdat)

        SeqsNew[n] = {
            'Time': History[0, :].tolist(),
            'Mark': History[1, :].tolist(),
            'Start': SeqsOld[n]['Stop'],
            'Stop': options['Tmax'],
        }

        index = np.where((SeqsOld[n]['Stop'] <= SeqsNew[n]['Time']) & (SeqsNew[n]['Time'] <= options['Tmax']))
        SeqsNew[n]['Time'] = np.array(SeqsNew[n]['Time'])[index].tolist()
        SeqsNew[n]['Mark'] = np.array(SeqsNew[n]['Mark'])[index].tolist()

        if n % 10 == 0 or n == options['N']:
            print('#seq={}/{}, #event={}, time={:.2f}sec'.format(
                n, options['N'], len(SeqsNew[n]['Mark']), time.time() - tic))

    return SeqsNew