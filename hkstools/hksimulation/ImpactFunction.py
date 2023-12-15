import numpy as np
from .Kernel import Kernel



def ImpactFunction(u, dt, para):
    A = np.reshape(para['A'][u, :, :], (para['A'].shape[1], para['A'].shape[2]))
    basis = Kernel(dt, para)

    # 扩展 basis 为与 A 具有相同形状
    if len(para['landmark']) == 1:
        basis_extended = np.ones_like(A) * basis
            # 使用 np.multiply 对 A 和 basis 逐元素相乘
        phi = np.multiply(A, basis_extended)
    else:
        basis_extended = basis
        phi = np.dot(A.T, basis.T)

    #phi = np.dot(A.T, basis.flatten())
    return phi