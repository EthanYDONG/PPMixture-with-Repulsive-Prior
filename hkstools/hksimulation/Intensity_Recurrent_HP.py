import numpy as np

def Intensity_Recurrent_HP(t_current, event_current, t_old, lambdat, para):
    """
    Compute the intensity functions of Hawkes processes recurrently according
    to historical intensity and current event.

    Parameters:
    - t_current: current time
    - event_current: current event
    - t_old: historical time
    - lambdat: historical intensity
    - para: parameters of Hawkes processes
        - para.mu: base exogenous intensity
        - para.A: coefficients of impact function
        - para.kernel: 'exp', 'gauss'
        - para.w: bandwidth of kernel
    
    Returns:
    - lambda_: intensity function of Hawkes process
    """
    dt = t_current - t_old
    weight = np.exp(-para['w'] * dt)
    lambda_ = para['mu'] + (lambdat - para['mu']) * weight

    if event_current:
        lambda_ += np.reshape(para['A'][event_current, 0, :], (para['A'].shape[2], 1))

    return lambda_

# eg：
# t_current = 2.0
# event_current = 1
# t_old = 1.0
# lambdat = np.array([0.5, 0.6, 0.7]).reshape(-1, 1)
# para = {'mu': np.array([0.2, 0.3, 0.4]).reshape(-1, 1),
#         'A': np.random.rand(3, 1, 3),
#         'kernel': 'exp',
#         'w': 0.1}
# result = Intensity_Recurrent_HP(t_current, event_current, t_old, lambdat, para)
# print(result)