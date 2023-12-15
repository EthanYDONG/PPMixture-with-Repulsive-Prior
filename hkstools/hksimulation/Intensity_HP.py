import numpy as np
from .Kernel import Kernel

def Intensity_HP(t, History, para):
    """
    Compute the intensity functions of Hawkes processes
    
    Parameters:
    - t: current time
    - History: historical event records
    - para: parameters of Hawkes processes
        - para.mu: base exogenous intensity
        - para.A: coefficients of impact function
        - para.kernel: 'exp', 'gauss'
        - para.w: bandwidth of kernel
    
    Returns:
    - lambda: intensity function of Hawkes process
    """
    lambda_ = para['mu'].flatten()

    if History.size > 0:
        Time = History[0, :]
        index = Time <= t
        Time = Time[index]
        Event = History[1, index]

        basis = Kernel(t - Time, para)
        A = para['A'][Event, :, :]

        for c in range(para['A'].shape[2]):
            lambda_[c] += np.sum(np.sum(basis * A[:, :, c]))

    lambda_ = lambda_ * (lambda_ > 0)

    return lambda_

