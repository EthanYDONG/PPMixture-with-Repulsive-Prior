import numpy as np
from .Intensity_Recurrent_HP import Intensity_Recurrent_HP

def SimulationFast_Thinning_ExpHP(para, options):
    """
    The fast simulation of Hawkes processes with exponential kernels
    Reference: Dassios, Angelos, and Hongbiao Zhao.
    "Exact simulation of Hawkes process with exponentially decaying intensity."
    Electronic Communications in Probability 18.62 (2013): 1-13.
    Provider: Hongteng Xu @ Georgia Tech, June 13, 2017
    """

    Seqs = {'Time': [], 'Mark': [], 'Start': [], 'Stop': [], 'Feature': []}

    for n in range(1, options['N'] + 1):
        t = 0
        History = np.array([]).reshape(2, 0)

        lambdat = para['mu']
        mt = np.sum(lambdat)

        while t < options['Tmax'] and History.shape[1] < options['Nmax']:
            s = np.random.exponential(1 / mt)
            U = np.random.rand()

            lambda_ts = Intensity_Recurrent_HP(t + s, None, t, lambdat, para)
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

                lambdat = Intensity_Recurrent_HP(t + s, [index], t, lambdat, para)
                t = t + s
                History = np.column_stack((History, [t, index]))

            mt = np.sum(lambdat)

        Seqs['Time'].append(History[0, :])
        Seqs['Mark'].append(History[1, :])
        Seqs['Start'] = 0
        Seqs['Stop'] = options['Tmax']

        index = np.where((Seqs['Time'][n - 1] <= options['Tmax']))[0]
        Seqs['Time'][n - 1] = Seqs['Time'][n - 1][index]
        Seqs['Mark'][n - 1] = Seqs['Mark'][n - 1][index]

        if n % 10 == 0 or n == options['N']:
            print(f'#seq={n}/{options["N"]}, #event={len(Seqs["Mark"][n - 1])}')

    return Seqs
