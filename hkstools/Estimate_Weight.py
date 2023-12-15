import numpy as np
from .DistanceSum_MPP import DistanceSum_MPP
def Estimate_Weight(configure, Seqs):
    # initialization
    configure['weight'] = np.random.rand(len(configure['id']))
    tau = configure['tau']
    obj = np.zeros(configure['epoch'] * len(Seqs))

    tic = 0
    for n in range(1, configure['epoch'] + 1):
        ind = np.random.permutation(len(Seqs))
        lr = configure['lr'] * (0.9)**(n - 1)
        for m in range(1, len(Seqs) + 1):
            X = np.vstack([Seqs[ind[m - 1]]['Time'], Seqs[ind[m - 1]]['Mark']])
            Prob, Delta = ChosenProbability(X, configure)

            grad = 0
            for t1 in range(1, Prob.shape[0] - tau + 1):
                for t2 in range(1, Prob.shape[1] - tau + 1):
                    if t1 != t2:
                        obj[(n - 1) * len(Seqs) + m - 1] += (
                            Prob[t1 - 1, t2 - 1] * Prob[t1 + tau - 1, t2 + tau - 1]
                        )
                        grad += (
                            2
                            * Prob[t1 - 1, t2 - 1]
                            * Prob[t1 + tau - 1, t2 + tau - 1]
                            * (Delta[:, t1 - 1, t2 - 1] + Delta[:, t1 + tau - 1, t2 + tau - 1])
                        )
            configure['weight'] -= lr * grad

            print(
                f'epoch={n}, #seq={m}/{len(Seqs)}, obj={obj[(n-1)*len(Seqs)+m-1]}, ||grad||={np.linalg.norm(grad):.4f}, time={tic:.2f}sec'
            )
    return configure, obj
# Replace ChosenProbability with your Python equivalent

# Replace ChosenProbability with your Python equivalent
def ChosenProbability(X, configure):
    # Implement ChosenProbability function here
    # This function should return Prob and Delta
    L = X.shape[1]
    W = configure['W']
    Prob = np.zeros((L - W + 1, L - W + 1))
    Dis = np.zeros_like(Prob)
    dis = np.zeros((len(configure['weight']), L - W + 1, L - W + 1))
    Delta = np.zeros_like(dis)

    for t1 in range(L - W + 1):
        for t2 in range(L - W + 1):
            if t1 != t2:
                X1 = X[:, t1:t1 + W]
                X2 = X[:, t2:t2 + W]
                Dis[t1, t2], dis[:, t1, t2] = DistanceSum_MPP(X1, X2, configure)
                Prob[t1, t2] = np.exp(-Dis[t1, t2] ** 2)

    Prob = Prob / (np.sum(Prob, axis=1)[:, np.newaxis] + np.finfo(float).eps)

    for t1 in range(L - W + 1):
        for t2 in range(L - W + 1):
            if t1 != t2:
                Delta[:, t1, t2] = np.sum(np.expand_dims(Prob[t1, :] * Dis[t1, :], axis=0) * dis[:, :, t1], axis=1) \
                                   - (Prob.shape[0] - 1) * Dis[t1, t2] * dis[:, t1, t2]

    return Prob, Delta
