import numpy as np
from scipy.sparse import csr_matrix

# Seqs：输入的序列集合。
# ClusterNum：聚类的数量。
# baseType、bandwidth、landmark：这些是可选参数，用于定义模型中的不同特征和属性。

def Initialization_Cluster_Basis(Seqs, ClusterNum, baseType=None, bandwidth=None, landmark=None):
    
    N = len(Seqs)
    D = np.zeros(N)

    for i in range(N):
        D[i] = np.max(Seqs[i]['Mark'])
    
    D = int(np.max(D))+1
    
    model = {'K': ClusterNum, 'D': D}

    #如果只传入 Seqs 和 ClusterNum，则使用默认的高斯核，计算标准差和最大时间以初始化模型。
    if baseType is None and bandwidth is None and landmark is None:
        sigma = np.zeros(N)
        Tmax = np.zeros(N)

        for i in range(N):
            sigma[i] = ((4 * np.std(Seqs[i]['Time'])**5) / (3 * len(Seqs[i]['Time'])))**0.2
            Tmax[i] = Seqs[i]['Time'][-1] + np.finfo(float).eps
        Tmax = np.mean(Tmax)

        model['kernel'] = 'gauss'#核函数类型
        model['w'] = np.mean(sigma)#带宽
        model['landmark'] = model['w'] * np.arange(0, np.ceil(Tmax / model['w']))#地标
        
    #如果传入更多参数，模型将根据这些参数进行初始化
    # 当 baseType 被提供，但 bandwidth 和 landmark 未提供时
    elif baseType is not None and bandwidth is None and landmark is None:
        model['kernel'] = baseType
        model['w'] = 1
        model['landmark'] = 0
    # 当 baseType 和 bandwidth 被提供，但 landmark 未提供时
    elif baseType is not None and bandwidth is not None and landmark is None:
        model['kernel'] = baseType
        model['w'] = bandwidth
        model['landmark'] = 0
    # 当所有参数都被提供时，这是最灵活的情况，允许用户完全自定义模型的初始化
    else:
        model['kernel'] = baseType if baseType is not None else 'gauss'
        model['w'] = bandwidth if bandwidth is not None else 1
        model['landmark'] = landmark if landmark is not None else 0

    model['alpha'] = 1
    M = len(model['landmark'])
    model['beta'] = np.ones((D, M, model['K'], D)) / (M * D**2)

    model['b'] = np.ones((D, model['K'])) / D

    
    # Initialize label and responsibility randomly
    label = np.ceil(model['K'] * np.random.rand(1, N)).astype(int)
    model['label']=label
    model['R'] = csr_matrix((np.ones(N), (np.arange(N), label.flatten()-1)), shape=(N, model['K'])).toarray()

    return model