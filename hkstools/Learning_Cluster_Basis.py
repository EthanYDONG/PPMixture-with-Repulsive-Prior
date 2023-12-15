import numpy as np
from .hksimulation.Kernel import Kernel
from .hksimulation.Kernel_Integration import Kernel_Integration
from scipy.special import psi
import time



#基于变分推断的学习过程，用于混合霍克斯过程模型
def Learning_Cluster_Basis(Seqs, model, alg):
    NLL = np.zeros(alg['outer'])#初始化一个记录负对数似然（Negative Log-Likelihood, NLL）的数组
    tic = time.time()
    for o in range(1, alg['outer']+1):
        # M步骤：通过调用 Maximization_MixHP 函数，这一步骤旨在最大化模型参数，给定当前对隐变量的估计
        model, NLL[o-1] = Maximization_MixHP(Seqs, model, alg) #更新 μ 和 A  #model['beta'] ，model['b'] 
        # E步骤：通过调用 Expectation_MixHP 函数，这一步骤基于当前模型参数来更新对隐变量的估计
        model,q_pai = Expectation_MixHP(Seqs, model, alg)#更新 π—— 和责任矩阵 r nk  #model['R']
        q_Z=model['R']#计算每个数据点属于不同簇的责任model['R']来近似
        print(f'MixMHP: Iter={o}, Obj={NLL[o-1]}, Time={time.time()-tic} sec')
    #将 NLL 数组存储在模型字典中并返回更新后的模型。
    model['NLL'] = NLL
    return model



def Maximization_MixHP(Seqs, model, alg):
    # given responsibility, calculate the expected number of sequences belonging
    # to the k-th cluster
    EX = model['R']
    # update parameters of Hawkes processes (mu_k, A_k), k=1,...,N
    # initialize
    A = model['beta']
    mu = np.sqrt(np.pi/2) * model['b']
    NLL = 0
    
    for in_ in range(alg['inner']):

        tmp1 = A.flatten() / model['beta'].flatten()
        tmp1[np.isnan(tmp1)] = 0
        tmp1[np.isinf(tmp1)] = 0
        
        tmp2 = mu.flatten()**2 / (2 * model['b'].flatten()**2)
        tmp2[np.isnan(tmp2)] = 0
        tmp2[np.isinf(tmp2)] = 0

        tmp3 = np.log(mu.flatten())
        tmp3[np.isnan(tmp3)] = 0
        tmp3[np.isinf(tmp3)] = 0

        NLL = np.sum(tmp1) + np.sum(tmp2) - np.sum(tmp3)

        MuA = 1.0 / (model['b']**2)
        MuA[np.isinf(MuA)] = 0
        MuB = 0
        MuC = -np.ones(model['b'].shape)

        AB = np.zeros(A.shape)
        AA = 1.0 / model['beta']
        AA[np.isinf(AA)] = 0
        # E-step: evaluate the responsibility using the current parameters
        for c, seq in enumerate(Seqs):
            Time = seq['Time']
            Event = seq['Mark'].astype(int)
            Tstart = seq['Start']
            # 找终止时间
            if 'Tmax' in alg and alg['Tmax'] is not None and alg['Tmax']:# 确保 alg['Tmax'] 不是空的，并且是一个数值
                Tstop = alg['Tmax']
                # 使用 np.less 来避免形状不匹配的问题
                valid_indices = np.less(Time, alg['Tmax'])
                Time = Time[valid_indices]
                Event = Event[valid_indices]
            else:
                Tstop = seq['Stop']
                
            N = len(Time)
            G = Kernel_Integration(Tstop - Time, model)

            TMPAA = np.zeros(A.shape)
            TMPAB = np.zeros(A.shape)
            TMPMuC = np.zeros(mu.shape)
            
            LL = 0
            
            for i in range(N):
                ui = Event[i]
                ti = Time[i]
                # 调整 np.tile 的使用以确保形状匹配
                G_expanded = np.tile(G[i][:, np.newaxis, np.newaxis], (1, model['D'], model['K'])).transpose(0, 2, 1)
                TMPAA[ui,:] += G_expanded     
                lambdai = mu[ui].reshape(1, -1) + np.finfo(float).eps
                pii = np.copy(lambdai)

                if i > 0:
                    tj = Time[:i]
                    uj = Event[:i]
                    gij = Kernel(ti - tj, model)
                    auiuj = A[uj, :, :, ui]
                    # 重塑 gij 以匹配 auiuj 的形状，重复model['K']次
                    pij = gij.reshape(gij.shape[0], gij.shape[1], 1).repeat(model['K'], axis=2) * auiuj  
       
                    tmp = np.sum(pij, axis=(0, 1))
                    lambdai += tmp.reshape(1, -1)

                    pij /= np.tile(lambdai.reshape(1, 1, model['K']), (pij.shape[0], pij.shape[1], 1))
                    for j in range(i):
                        uj = Event[j]
                        TMPAB[uj, :, :, ui] -= pij[j, :, :]
                        
                LL = LL + np.log(lambdai)
                pii = pii / lambdai
                TMPMuC[ui] -= pii.flatten()

            LL = LL - (Tstop - Tstart) * np.sum(mu)

            tmp = np.sum(np.sum(G.reshape(G.shape[0], G.shape[1], 1).repeat(model['K'], axis=2) * np.sum(A[Event, :, :, :], axis=3), axis=1), axis=0)
            LL = LL - tmp.ravel()

            MuB = MuB + (Tstop - Tstart) * EX[c, :]#EX[c,:] 表示第 c 个数据点属于每个簇的概率。
            for k in range(model['K']):
                AA[:, :, k] = AA[:, :, k] + EX[c, k] * TMPAA[:, :, k]
                AB[:, :, k] = AB[:, :, k] + EX[c, k] * TMPAB[:, :, k]
                MuC[:, k] = MuC[:, k] + EX[c, k] * TMPMuC[:, k]

            NLL = NLL - sum(EX[c, :]* LL.ravel())

        MuBB = np.tile(MuB, (model['D'], 1))


        mutmp = (-MuBB + np.sqrt(MuBB**2 - 4 * MuA * MuC)) / (2 * MuA)
        Atmp = -AB / AA

        Atmp[np.isnan(Atmp)] = 0
        Atmp[np.isinf(Atmp)] = 0
        mutmp[np.isnan(mutmp)] = 0
        mutmp[np.isinf(mutmp)] = 0

        Err = np.sum(np.abs(A.flatten() - Atmp.flatten())) / np.sum(np.abs(A.flatten()))
        print(f'Inner= {in_}, Obj={NLL}, RelErr={Err}')

        A = Atmp
        mu = mutmp
        if Err < alg['thres'] or in_ == alg['inner']:
            break

    model['beta'] = A
    model['b'] = np.sqrt(2/np.pi) * mu
    return model,NLL




def Expectation_MixHP(Seqs, model, alg):
    Nk = np.sum(model['R'], axis=0)  # 10.51
    alpha = model['alpha'] + Nk  # Dirichlet
    
    Elogpi = psi(alpha) - psi(np.sum(alpha))  # 10.66
    # calculate responsibility
    EX = np.zeros((len(Seqs), model['K']))
    # E-step: evaluate the responsibility using the current parameters
    for c in range(len(Seqs)):
        Time = Seqs[c]['Time']
        Event = Seqs[c]['Mark']
        Tstart = Seqs[c]['Start']

        if not alg['Tmax']:
            Tstop = Seqs[c]['Stop']
        else:
            Tstop = alg['Tmax']
            indt = Time < alg['Tmax']
            Time = Time[indt]
            Event = Event[indt]
        N = len(Time)
        # calculate the integral decay function in the log-likelihood function
        G = Kernel_Integration(Tstop - Time, model)
        LL = Elogpi
        for i in range(N):
            ui = Event[i]
            ti = Time[i]
            Elambdai = np.sqrt(np.pi/2) * model['b'][int(ui), :] + np.finfo(float).eps
            Vlambdai = (2 - np.pi/2) * (model['b'][int(ui), :])**2

            if i > 0:
                tj = Time[:i]
                uj = Event[:i]

                gij = Kernel(ti - tj, model)

                uj_int = uj.astype(int) if isinstance(uj, np.ndarray) else int(uj)
                ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)

                auiuj = model['beta'][uj_int, :, :, ui_int]
                pij = np.repeat(gij[:, :, np.newaxis], 2, axis=2)* auiuj
 
                tmp = np.sum(pij, axis=(0,1)).reshape(1,-1)
                Elambdai = Elambdai + tmp
                
                tmp = np.sum(pij**2, axis=(0,1)).reshape(1,-1)#np.sum(np.sum(pij**2, 1), 2)
                Vlambdai = Vlambdai + tmp.ravel()

            LL = LL + np.log(Elambdai) - Vlambdai / (2 * Elambdai**2)

        LL = LL - (Tstop - Tstart) * np.sqrt(np.pi/2) * np.sum(model['b'])
        
        # 第一步：重复 G 数组
        repeated_G = np.repeat(G[:, :, np.newaxis], 2, axis=2)
        # 第二步：对 model.beta 在第四维度（axis=3）进行求和
        Event_int = Event.astype(int) if isinstance(Event, np.ndarray) else int(Event)
        summed_beta = np.sum(model['beta'][Event_int, :, :, :], axis=3)
        # 第三步：计算 repeated_G 和 summed_beta 的逐元素乘积

        
        elementwise_product = repeated_G * summed_beta
        # 第四步：在第一维度（axis=0）上求和
        sum_along_first_dim = np.sum(elementwise_product, axis=0)
        # 第五步：在第二维度（axis=1）上求和
        tmp = np.sum(sum_along_first_dim[np.newaxis, :, :], axis=1)
  
        LL = LL - tmp.ravel()

        XX = (LL - np.max(LL))
        EX[c, :] = (np.exp(XX)) / np.sum(np.exp(XX))

    model['R'] = EX
    return model,Elogpi