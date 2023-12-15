import numpy as np

options = {
    'N' : 100,
    'Nmax' : 100,
    'Tmax' : 50,
    'tstep' : 0.1,
    'dt' : 0.1,
    'M' : 250,
    'GenerationNum' : 10
}

global_config = {
    'D' : 3, # dimension of Hwkers process
    'K' : 2, # the number of clusters'
    'nTest' : 5,
    'nSeg' : 5
}

def get_paras():
    paras = []
    D = global_config['D']
    # the number of para shoule be the same as k
    landmark = [0]
    L = len(landmark)
    para1={
        'kernel':'exp',
        'landmark':landmark,
        'mu':np.random.rand(D,1)/D,
        'A':np.zeros(shape=(D,D,L)),
        'w':0.5,
        'A_para1':0.7,
        'A_para2':0.5
    }
    for l in range(1,L+1):
        para1['A'][:,:,l-1] = (para1['A_para1']**l)*np.random.rand(D, D)
    max_eig = max(np.abs(np.linalg.eigvals(para1['A'][:, :, 0])))
    para1['A'] = para1['A_para2'] * para1['A'] / max_eig
    paras.append(para1)

    landmark = np.arange(0,13,3)
    L = len(landmark)
    para2={
        'kernel':'gauss',
        'landmark':landmark,
        'mu':np.random.rand(D,1)/D,
        'A':np.zeros(shape=(D,D,L)),
        'w':1,
        'A_para1':0.9,
        'A_para2':0.25
    }
    for l in range(1,L+1):
        para2['A'][:,:,l-1] = (para2['A_para1']**l)*np.random.rand(D, D)
    max_eig = max(np.abs(np.linalg.eigvals(np.sum(para2['A'], axis=2))))
    para2['A'] = para2['A_para2'] * para2['A'] / max_eig
    para2['A'] = np.reshape(para2['A'], (D, L, D))
    paras.append(para2)
    
    return paras
# class para:
#     def __init__(self,kernel = 'exp',landmark = 0,w=1,coefficents=None
#                  ,global_config = None) -> None:
#         if not global_config or not coefficents:
#             print('global_config empty')
#             return 
#         self.kernel = kernel
#         self.landmark = landmark
#         D = global_config['D']
#         L = len(self.landmark)
#         self.mu = np.random.random(D).reshape((D,1))/D
#         self.A = np.zeros(shape = (D,D,L))
#         self.coefficents = coefficents
#         for l in range(1,L+1):
#             self.A[:,:,l-1] = (self.coefficent[0]**l)*np.random.random(size= (D,D))
#         #self.A = self.coefficents[1] * self.A./max(abs(e))
#         max_eig = max(np.abs(np.linalg.eigvals(self.A.reshape(-1,D,D))))
#         self.A = self.coefficents[1]*self.A/max_eig
#         self.A = self.A.reshape((D,L,D))
#         self.w = w
