import numpy as np
import matplotlib.pyplot as plt
import time
#模拟霍克斯过程。“Branch”可能暗示了它使用了某种分支过程方法来模拟霍克斯过程。
from .hksimulation.Simulation_Branch_HP import Simulation_Branch_HP
#初始化一种基于霍克斯过程的聚类模型
from Initialization_Cluster_Basis import Initialization_Cluster_Basis
#学习或训练上述提到的霍克斯过程聚类模型
from Learning_Cluster_Basis import Learning_Cluster_Basis
#估计某些类型的权重，这些权重在霍克斯过程或相关的点过程模型中很重要
from Estimate_Weight import Estimate_Weight,DistanceSum_MPP
from Loglike_Basis import Loglike_Basis

#1. 参数设置
options = {
    'N': 100, 'Nmax': 100, 'Tmax': 50, 'tstep': 0.1,
    'dt': [0.1], 'M': 250, 'GenerationNum': 10
}
D = 3
K = 2
nTest = 5
nSeg = 5
nNum = options['N'] / nSeg

print('Approximate simulation of Hawkes processes via branching process')


########################################################################################

#2. 霍克斯过程的模拟（通过一个分支过程近似地模拟霍克斯过程）
# First cluster: Hawkes process with exponential kernel
print('Simple exponential kernel')
para1 = {'kernel': 'exp', 'landmark': [0]}
para1['mu'] = np.random.rand(D) / D
L = len(para1['landmark'])
para1['A'] = np.zeros((D, D, L))
for l in range(1, L + 1):
    para1['A'][:, :, l - 1] = (0.7**l) * np.random.rand(D,D)

#import pdb;pdb.set_trace()
# 对每个切片应用 np.linalg.eigh
eigvals_list = []
eigvecs_list = []
for l in range(L):
    eigvals, eigvecs = np.linalg.eigh(para1['A'][:, :, l])
    eigvals_list.append(eigvals)
    eigvecs_list.append(eigvecs)

# 对所有特征值进行处理
all_eigvals = np.concatenate(eigvals_list)
max_eigval = np.max(all_eigvals)
#import pdb;pdb.set_trace()
# 使用最大特征值进行归一化
para1['A'] = 0.5 * para1['A'] / max_eigval
para1['w'] = 0.5

Seqs1 = Simulation_Branch_HP(para1, options)


########################################################################################

# Second cluster: Hawkes process with Gaussian kernel
print('Complicated gaussian kernel')
para2 = {'kernel': 'gauss', 'landmark': np.arange(0, 13, 3)}
para2['mu'] = np.random.rand(D) / D
L = len(para2['landmark'])
para2['A'] = np.zeros((D, D, L))
for l in range(1, L + 1):
    para2['A'][:, :, l - 1] = (0.9**l) * np.random.rand(D,D)
para2['A'] = 0.25 * para2['A'] / np.max(np.abs(np.linalg.eigh(np.sum(para2['A'], axis=2))[0]))
para2['A'] = np.reshape(para2['A'], (D, L, D))
#import pdb;pdb.set_trace()
para2['w'] = 1
Seqs2 = Simulation_Branch_HP(para2, options)
SeqsMix = Seqs1 + Seqs2
#import pdb;pdb.set_trace()

# Ground truth: the similarity matrix of event sequences.
# GT[i, j] = 1 if the sequence i and the sequence j belong to the same cluster


GT = np.block([[np.ones((options['N'], options['N'])),  np.zeros((options['N'], options['N']))],
               [ np.zeros((options['N'], options['N'])), np.ones((options['N'], options['N']))]])
# #3. 霍克斯过程的混合模型学习
# 初始化模型，并通过某种学习算法（由 alg 参数指定）来学习霍克斯过程的混合模型。
# 最后，计算得到一个估计的相似度矩阵 Est1。

# Learning a mixture model of Hawkes processes
# initialize
model = Initialization_Cluster_Basis(SeqsMix, 2)
alg = {'outer': 8, 'rho': 0.1, 'inner': 5, 'thres': 1e-5, 'Tmax': []}#'inner': 5

model = Learning_Cluster_Basis(SeqsMix, model, alg)

Loglikes=Loglike_Basis(SeqsMix, model, alg)



labels1 = np.argmax(model['R'], axis=1)
# 估计相似性矩阵
Est1 = np.zeros((len(labels1), len(labels1)), dtype=int)
for i in range(len(labels1)):
    for j in range(len(labels1)):
        if labels1[i] == labels1[j]:
            Est1[i, j] = 1


# #4. 标记点过程的距离度量学习:
# 配置了一些参数（如时间窗口大小，维度等），并用这些参数来估计权重。
# 计算序列之间的距离，并基于这些距离构建另一个估计的相似度矩阵 SimilarMat。

# Learning a distance metric of marked point processes
configure = {'M': [options['Tmax'], D], 'id': [1, 2, [1, 2]], 'tau': 1, 'W': 5, 'epoch': 10, 'lr': 1e-4}
configure, obj = Estimate_Weight(configure, SeqsMix)


Dis = np.zeros((len(SeqsMix), len(SeqsMix)))
tic = time.time()
for n in range(len(SeqsMix) - 1):
    Xn = np.vstack([SeqsMix[n]['Time'], SeqsMix[n]['Mark']])#concatenate
    for m in range(n + 1, len(SeqsMix)):
        Xm = np.vstack([SeqsMix[m]['Time'], SeqsMix[m]['Mark']])

        Dis[n, m], _ = DistanceSum_MPP(Xn, Xm, configure)
toc = time.time()
Dis = Dis + Dis.T
# estimate similarity matrix
SimilarMat = np.exp(-Dis**2 / (2 * np.var(Dis)))


# Visualize clustering results
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(GT, cmap='gray', interpolation='none')
plt.title('Ground truth of clusters')
plt.subplot(1, 3, 2)
plt.imshow(Est1, cmap='gray', interpolation='none')
plt.title('Mixture of Hawkes processes')
plt.subplot(1, 3, 3)
plt.imshow(SimilarMat, cmap='gray', interpolation='none')
plt.title('Distance metrics of MPP')
plt.show()