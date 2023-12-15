
import numpy as np
def ImpactFunction(u, dt, para):
    row = para['A'].shape[1]
    col = para['A'].shape[2]
    A = para['A'][u,:,:].reshape((row,col))
    basis = Kernel(dt,para)
    basis = basis.flatten()
    phi = np.dot(A.T,basis)
    return phi


def Kernel(dt, para):
    landmark = para['landmark']
    distance = np.tile(dt[:, np.newaxis], (1, len(landmark)))
    - np.tile(landmark[np.newaxis, :], (len(dt), 1))
    
    kernel = para['kernel']
    if kernel == 'exp':
        g = para['w']*np.exp(-para['w']*distance)
        g [g > 1] = 0
    elif kernel == 'gauss':
        g = np.exp(-(distance**2) / (2*para['w']**2)) / (np.sqrt(2 * np.pi) * para['w'])
    
    else:
        print('kernel wrong')
    return g