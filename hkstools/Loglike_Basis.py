import numpy as np
from .Kernel_Integration import Kernel_Integration
from .hksimulation.Kernel import Kernel
# import sys
# print(sys.path)


def Loglike_Basis(Seqs, model, alg):#霍克斯过程模型的负对数似然
    Aest = model['beta']#A
    muest = model['b']#mu
    Loglikes = [] 

    for c in range(len(Seqs)):
        Time = Seqs[c]['Time']
        Event = Seqs[c]['Mark']
        Event_int = Event.astype(int) if isinstance(Event, np.ndarray) else int(Event)
        Tstart = Seqs[c]['Start']

        if not alg['Tmax']:
            Tstop = Seqs[c]['Stop']
        else:
            Tstop = alg['Tmax']
            indt = Time < alg['Tmax']
            Time = Time[indt]
            Event = Event[indt]

        dT = Tstop - Time
        GK = Kernel_Integration(dT, model)
        

        Nc = len(Time)
        Loglike = 0
        for i in range(Nc):
            ui = Event[i]
            ti = Time[i]
            ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
            lambdai = muest[ui_int]#基于当前模型参数的条件强度函数，给定历史事件，该事件发生的概率


            if i > 0:
                tj = Time[:i]
                uj = Event[:i]
                uj_int = uj.astype(int) if isinstance(uj, np.ndarray) else int(uj)
                ui_int = ui.astype(int) if isinstance(ui, np.ndarray) else int(ui)
                dt = ti - tj
                gij = Kernel(dt, model)
                auiuj = Aest[uj_int, :,:, ui_int]
                pij = np.repeat(gij[:, :, np.newaxis], 2, axis=2)* auiuj
                #pij = auiuj * gij
                lambdai = lambdai + np.sum(pij, axis=(0,1)).reshape(1,-1)#np.sum(pij)

            Loglike = Loglike - np.log(lambdai)
        
        Loglike = Loglike + (Tstop - Tstart) * np.sum(muest)
    
        
        # GK_reshape=np.repeat(GK[:, :, np.newaxis], Aest.shape[0], axis=2)
        # Loglike = Loglike + np.sum(np.sum(GK_reshape * np.sum(Aest[Event_int, :, :, :], axis=2)))
        GK_reshape=np.repeat(np.repeat(GK[:, :, np.newaxis, np.newaxis], Aest.shape[2], axis=2), Aest.shape[3], axis=3)#np.repeat(GK[:, :, np.newaxis], Aest.shape[0], axis=2)
        Loglike = Loglike + (GK_reshape * Aest[Event_int, :, :, :]).sum(axis=(0, 1, 3)).reshape(1, 2)
        # print(GK.shape)
        # print(GK[:, :, np.newaxis, np.newaxis].shape)
        print(GK_reshape[0,0,:,:])
        print(np.repeat(np.repeat(GK[:, :, np.newaxis, np.newaxis], Aest.shape[2], axis=2), Aest.shape[3], axis=3).shape)
        # print(Aest.shape)
        if model['label'][0][c]==1:
            Loglikes.append(-Loglike[0][0])
        elif model['label'][0][c]==2:
            Loglikes.append(-Loglike[0][1])
        else:
            Loglikes.append(666666)
        
    return Loglikes