import numpy as np

def Infectivity_TVHP(T, t, Period, Shift, MaxInfect, Type):
    """
    Generate infectivity matrix A(t) in R^{U*U}
    
    Parameters:
    - T: the time interval
    - t: current time
    - Period: predefined period parameter in R^{U*U}
    - Shift: predefined time-shift parameter in R^{U*U}
    - MaxInfect: maximum infectivity
    - Type: infectivity mode type, can be 1 or 2
    
    Returns:
    - A: infectivity matrix A(t) in R^{U*U}
    """
    if Type == 1:
        M = np.round((Period / T) * (t - Shift))
        A = MaxInfect * 0.5 * (1 - (-1) ** M)
    elif Type == 2:
        A = MaxInfect * 0.5 * (1 + np.cos((2 * Period / T) * t - Shift))
    else:
        raise ValueError("Invalid Type value. Type should be 1 or 2.")
    print('A:   ',A)
    return A