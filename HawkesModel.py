import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import skewnorm
from scipy.stats import norm
from scipy.sparse import csr_matrix
from hkstools.hksimulation.Simulation_Branch_HP import Simulation_Branch_HP
from hkstools.Initialization_Cluster_Basis import Initialization_Cluster_Basis
from hkstools.Learning_Cluster_Basis import Learning_Cluster_Basis
from hkstools.Estimate_Weight import Estimate_Weight
from hkstools.Loglike_Basis import Loglike_Basis
from hkstools.DistanceSum_MPP import DistanceSum_MPP
from hkstools.Kernel_Integration import Kernel_Integration
from hkstools.hksimulation.Kernel import Kernel
from scipy.special import erf

class HawkesModel:
    def __init__(self, Seqs, ClusterNum, Tmax):
        self.Seqs = Seqs
        self.clusternum = ClusterNum
        self.model = None  # Initialize model as None
        self.Tmax = Tmax

    def Initialization_Cluster_Basis(self, baseType=None, bandwidth=None, landmark=None):
        N = len(self.Seqs)
        D = np.zeros(N)
        for i in range(N):
            D[i] = np.max(self.Seqs[i]['Mark'])
        D = int(np.max(D)) + 1
        self.K = self.clusternum
        self.D = D

        # 如果只传入 Seqs 和 ClusterNum，则使用默认的高斯核，计算标准差和最大时间以初始化模型。
        if baseType is None and bandwidth is None and landmark is None:
            sigma = np.zeros(N)
            Tmax = np.zeros(N)

            for i in range(N):
                sigma[i] = ((4 * np.std(self.Seqs[i]['Time'])**5) / (3 * len(self.Seqs[i]['Time'])))**0.2
                Tmax[i] = self.Seqs[i]['Time'][-1] + np.finfo(float).eps
            Tmax = np.mean(Tmax)

            self.kernel = 'gauss'  # 核函数类型
            self.w = np.mean(sigma)  # 带宽
            self.landmark = self.w * np.arange(0, np.ceil(Tmax / self.w))  # 地标

        # 如果传入更多参数，模型将根据这些参数进行初始化
        # 当 baseType 被提供，但 bandwidth 和 landmark 未提供时
        elif baseType is not None and bandwidth is None and landmark is None:
            self.kernel = baseType
            self.w = 1
            self.landmark = 0
        # 当 baseType 和 bandwidth 被提供，但 landmark 未提供时
        elif baseType is not None and bandwidth is not None and landmark is None:
            self.kernel = baseType
            self.w = bandwidth
            self.landmark = 0
        # 当所有参数都被提供时，这是最灵活的情况，允许用户完全自定义模型的初始化
        else:
            self.kernel = baseType if baseType is not None else 'gauss'
            self.w = bandwidth if bandwidth is not None else 1
            self.landmark = landmark if landmark is not None else 0

        self.alpha = 1
        self.M = len(self.landmark)
        self.beta_a = np.ones((D, self.M, self.K, D)) / (self.M * D**2)   # kernel前的系数 accidk D M K D
        self.beta_na = np.ones((D, self.M, 1, D)) / (self.M * D**2)
        # D = C   M = land_mark  basic function = D           K = cluster
        self.b_a = np.ones((D, self.K)) / D     # base intensity shape = [event_type , k]
        self.b_na = np.ones((D, 1))/D
        # Initialize label and responsibility randomly
        label = np.ceil(self.K * np.random.rand(1, N)).astype(int)-1
        self.label = label
        # self.R_a = csr_matrix((np.ones(N), (np.arange(N), label.flatten())), shape=(N, self.K)).toarray()
        # self.R_na = csr_matrix((np.ones(N), (np.arange(N), label.flatten())), shape=(N, 1)).toarray()
        self.R_a = None
        self.R_na = None
        # pi_k jump
    def Kernel(self,dt):
        dt = np.array(dt).flatten()
        # Create a 2D array of landmarks
        landmarks = np.array(self.landmark)[np.newaxis, :]
        # Tile dt to have the same number of columns as the number of landmarks
        dt_tiled = np.tile(dt[:, np.newaxis], (1, len(self.landmark)))
        distance = dt_tiled - landmarks
        if self.kernel == 'exp':
            g = self.w * np.exp(-self.w * distance)
            g[g > 1] = 0

        elif self.kernel == 'gauss':
            g = np.exp(-(distance**2) / (2 * self.w**2)) / (np.sqrt(2 * np.pi) * self.w)
        else:
            print('Error: please assign a kernel function!')
        return g    
            
    def Kernel_Integration(self,dt):
        dt = dt.flatten()
        distance = np.tile(dt[:, np.newaxis], (1, len(self.landmark))) - np.tile(self.landmark, (len(dt), 1))
        landmark = np.tile(self.landmark, (len(dt), 1))
        if self.kernel == 'exp':
            G = 1 - np.exp(-self.w * (distance - landmark))
            G[G < 0] = 0
        elif self.kernel == 'gauss':
            G = 0.5 * (erf(distance / (np.sqrt(2) * self.w)) + erf(landmark / (np.sqrt(2) * self.w)))
        else:
            print('Error: please assign a kernel function!')
            G = None
        return G

    def Loglike(self,cluster_idx,mu_prop = None,A_prop = None,index_prop = None):     
        if mu_prop is  None:
            muest = self.b_a  # mu   base intensity event_type * k
        else:
            muest = mu_prop  ##需要确认是否对cluster event_Typr * 1
        if A_prop is  None:
            Aest = self.beta_a # DMKD
        else:
            Aest =  A_prop# A   the weights of different kernels DMD
        if index_prop is  None:
            indexest = self.label
        else:
            indexest = index_prop

        label_k_indices = np.where(indexest == cluster_idx)[1]
        label_k_seqs = [self.Seqs[idx] for idx in label_k_indices]
        Loglikes = []
        for c in range(len(label_k_seqs)):
            Time = label_k_seqs[c]['Time']
            Event = label_k_seqs[c]['Mark']
            Event_int = Event.astype(int) if isinstance(Event, np.ndarray) else int(Event)
            Tstart = label_k_seqs[c]['Start']
            if not self.Tmax:
                Tstop = label_k_seqs[c]['Stop']
            else:
                Tstop = self.Tmax
                indt = Time < self.Tmax
                Time = Time[indt]
                Event = Event[indt]
            dT = Tstop - Time
            GK = self.Kernel_Integration(dT)            
            Nc = len(Time)
            Loglike = 0
            for i in range(Nc):
                ui = Event[i]
                ti = Time[i]
                ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                #import pdb;pdb.set_trace()
                lambdai = muest[ui_int][cluster_idx] if  mu_prop is None else muest[ui_int]
                if i > 0:
                    tj = Time[:i]
                    uj = Event[:i]
                    uj_int = uj.astype(int) if isinstance(uj, np.ndarray) else int(uj)
                    ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                    dt = ti - tj
                    gij = self.Kernel(dt)
                    auiuj = Aest[uj_int, :, cluster_idx, ui_int] if ( A_prop is None) else Aest[uj_int,:,ui_int]
                    #import pdb;pdb.set_trace()
                    pij = gij * auiuj
                    #import pdb;pdb.set_trace()
                    lambdai = lambdai + np.sum(pij)
             
                #if (((self.b_a<=0).any)<=0):
                    # print(muest)
                    # print(lambdai)
                    #print(self.b_a)
                Loglike = Loglike - np.log(lambdai)
            
            # print(Loglike)
            # import pdb;pdb.set_trace()
            Loglike = Loglike + (Tstop - Tstart) * np.sum(muest)
            GK_reshape = np.repeat(GK[:, :, np.newaxis], Aest.shape[3], axis=2) if (A_prop is None) else np.repeat(GK[:, :, np.newaxis], Aest.shape[2], axis=2)
            #import pdb;pdb.set_trace()
            Loglike = (Loglike + (GK_reshape * Aest[Event_int, :, cluster_idx, :]).sum()) if (A_prop is None) else (Loglike + (GK_reshape * Aest[Event_int, :, :]).sum())
            
            Loglikes.append(-Loglike)
        loglike_for_allseqinthiscluster = sum(Loglikes)
        return loglike_for_allseqinthiscluster
    

    def loglike_one_a(self,seq_one,cluster_idx):  
            Time = seq_one['Time']
            Event = seq_one['Mark']
            Event_int = Event.astype(int) if isinstance(Event, np.ndarray) else int(Event)
            Tstart = seq_one['Start']
            if not self.Tmax:
                Tstop = seq_one['Stop']
            else:
                Tstop = self.Tmax
                indt = Time < self.Tmax
                Time = Time[indt]
                Event = Event[indt]
            dT = Tstop - Time
            GK = self.Kernel_Integration(dT)            
            Nc = len(Time)
            Loglike = 0
            for i in range(Nc):
                ui = Event[i]
                ti = Time[i]
                ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                lambdai = self.b_a[ui_int][cluster_idx]
                if i > 0:
                    tj = Time[:i]
                    uj = Event[:i]
                    uj_int = uj.astype(int) if isinstance(uj, np.ndarray) else int(uj)
                    ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                    dt = ti - tj
                    gij = self.Kernel(dt)
                    auiuj = self.beta_a[uj_int, :, cluster_idx, ui_int]
                    pij = gij * auiuj
                    #import pdb;pdb.set_trace()
                    lambdai = lambdai + np.sum(pij)
                    
                Loglike = Loglike - np.log(lambdai)
            #import pdb;pdb.set_trace()
            # print(Loglike)
            Loglike = Loglike + (Tstop - Tstart) * np.sum(self.b_a)
            GK_reshape = np.repeat(GK[:, :, np.newaxis], self.beta_a.shape[3], axis=2)
            #import pdb;pdb.set_trace()
            Loglike = Loglike + (GK_reshape * self.beta_a[Event_int, :, cluster_idx, :]).sum()
            return -Loglike
    def loglike_one_na(self,seq_one,cluster_idx):  
            Time = seq_one['Time']
            Event = seq_one['Mark']
            Event_int = Event.astype(int) if isinstance(Event, np.ndarray) else int(Event)
            Tstart = seq_one['Start']
            if not self.Tmax:
                Tstop = seq_one['Stop']
            else:
                Tstop = self.Tmax
                indt = Time < self.Tmax
                Time = Time[indt]
                Event = Event[indt]
            dT = Tstop - Time
            GK = self.Kernel_Integration(dT)            
            Nc = len(Time)
            Loglike = 0
            for i in range(Nc):
                ui = Event[i]
                ti = Time[i]
                ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                # print(ui_int,'??') 
                lambdai = self.b_na[ui_int][cluster_idx] #event_type * 1
                if i > 0:
                    tj = Time[:i]
                    uj = Event[:i]
                    uj_int = uj.astype(int) if isinstance(uj, np.ndarray) else int(uj)
                    ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                    dt = ti - tj
                    gij = self.Kernel(dt)
                    auiuj = self.beta_na[uj_int, :, cluster_idx, ui_int]
                    pij = gij * auiuj
                    #import pdb;pdb.set_trace()
                    lambdai = lambdai + np.sum(pij)
                    
                Loglike = Loglike - np.log(lambdai)
            #import pdb;pdb.set_trace()
            Loglike = Loglike + (Tstop - Tstart) * np.sum(self.b_na)
            GK_reshape = np.repeat(GK[:, :, np.newaxis], self.beta_na.shape[3], axis=2)
            #import pdb;pdb.set_trace()
            Loglike = Loglike + (GK_reshape * self.beta_na[Event_int, :, cluster_idx, :]).sum()
            return -Loglike

    def update_model(self):
        pass



if __name__=='__main__':
        
    np.random.seed(0)

    ###################simulation hks data#######################
    options = {
        'N': 100, 'Nmax': 100, 'Tmax': 50, 'tstep': 0.1,
        'dt': [0.1], 'M': 250, 'GenerationNum': 10
    }
    D = 3   # event type个数
    K = 2  # cluster个数
    nTest = 5
    nSeg = 5
    nNum = options['N'] / nSeg
    mucenter = np.random.rand(D) / D
    mudelta = 0.05
    # First01 cluster: Hawkes process with exponential kernel

    print('01 Simple exponential kernel')
    para1 = {'kernel': 'exp', 'landmark': [0]}
    para1['mu'] = mucenter+mudelta
    L = len(para1['landmark'])
    para1['A'] = np.zeros((D, D, L))
    for l in range(1, L + 1):
        para1['A'][:, :, l - 1] = (0.7**l) * np.random.rand(D,D)
    eigvals_list = []
    eigvecs_list = []
    for l in range(L):
        eigvals, eigvecs = np.linalg.eigh(para1['A'][:, :, l])
        eigvals_list.append(eigvals)
        eigvecs_list.append(eigvecs)
    all_eigvals = np.concatenate(eigvals_list)
    max_eigval = np.max(all_eigvals)
    para1['A'] = 0.5 * para1['A'] / max_eigval
    para1['w'] = 0.5
    Seqs1 = Simulation_Branch_HP(para1, options)
    # First02 cluster: Hawkes process with exponential kernel
    print('02 Simple exponential kernel')
    para2 = {'kernel': 'exp', 'landmark': [0]}
    para2['mu'] = mucenter-mudelta
    L = len(para2['landmark'])
    para2['A'] = np.zeros((D, D, L))
    for l in range(1, L + 1):
        para2['A'][:, :, l - 1] = (0.7**l) * np.random.rand(D,D)
    eigvals_list = []
    eigvecs_list = []
    for l in range(L):
        eigvals, eigvecs = np.linalg.eigh(para2['A'][:, :, l])
        eigvals_list.append(eigvals)
        eigvecs_list.append(eigvecs)
    all_eigvals = np.concatenate(eigvals_list)
    max_eigval = np.max(all_eigvals)
    para2['A'] = 0.5 * para2['A'] / max_eigval
    para2['w'] = 0.5
    Seqs2 = Simulation_Branch_HP(para2, options)


    '''
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

    '''
    SeqsMix = Seqs1 + Seqs2
        
    cluster_num_init = 2
    landmark_num = 5
    hksmodel = HawkesModel(SeqsMix,cluster_num_init,options['Tmax'])
    hksmodel.Initialization_Cluster_Basis()
    print(hksmodel.Loglike(0))
    #import pdb;pdb.set_trace()
    #print(SeqsMix[0])