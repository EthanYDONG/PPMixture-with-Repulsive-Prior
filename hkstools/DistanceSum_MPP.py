import numpy as np

def DistanceSum_MPP(X, Y, configure): # 计算两个点过程实例 X 和 Y 之间的加权距离
    # Implement the DistanceSum_MPP function here
    # This function should return the distance and distance array between two sequences X1 and X2
    # based on the configuration parameters.
    dis = np.zeros(len(configure['weight']))
    DisS = 0

    for i in range(len(configure['weight'])):

        dis[i] = Distance_MPP(X, Y, configure['M'], configure['id'][i])
        DisS += configure['weight'][i] * dis[i]
    
    return DisS, dis

def Distance_MPP(X, Y, M, id):
    """
    Define the distance of two point processes' instances.
    """
    if isinstance(id, int):
        id = [id]
    dis = 0
    for n1 in range(X.shape[1]):
        for n2 in range(X.shape[1]):
            tmp = 1
            for i in range(len(id)):
                tmp *= M[i] - abs(X[i, n1] - X[i, n2])
                if tmp == 0:
                    break
            dis += tmp

    for n1 in range(Y.shape[1]):
        for n2 in range(Y.shape[1]):
            tmp = 1
            for i in range(len(id)):
                tmp *= M[i] - abs(Y[i, n1] - Y[i, n2])
                if tmp == 0:
                    break
            dis += tmp

    for n1 in range(X.shape[1]):
        for n2 in range(Y.shape[1]):
            tmp = 1
            for i in range(len(id)):
                tmp *= M[i] - abs(X[i, n1] - Y[i, n2])
                if tmp == 0:
                    break
            dis -= 2 * tmp

    dis = np.sqrt(abs(dis))
    return dis