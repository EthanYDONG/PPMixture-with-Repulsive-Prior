import numpy as np
import pandas as pd
import random
from Utils import *
from simulation_config import *

# def Seqs():

#     def __init(self,para=None):
#         self.Time = []
#         self.Mark = []
#         self.Start = []
#         self.Stop = []
#         self.Feature = []
#         self.para = para

class simulatino_HP():

    def __init__(self,options = None) -> None:
        
        self.options = options
    
    def Simulation_Thining_Poisson(self, mu, t_start, t_end):
        t = t_start
        History = []
        mt = sum(mu)
        while t < t_end:
            s = np.random.exponential(1/mt)[0]
            #print('s:',s)
            t = t + s
            #print('t:',t)
            u = np.random.rand()*mt
            sumIs = 0
            for d in range(len(mu)):
                sumIs += mu[d]
                if sumIs >= u:
                    break
            index = d
            
            History.append([t,index])
        #print(History)
        History = np.array(History).T
        index = np.where(History[0, :] < t_end)
        History = History[:, index[0]]
        return History

    def generation(self,para):
        Seqs = []
        for n in range(self.options['N']):
            History =  self.Simulation_Thining_Poisson(para['mu'], 0, self.options['Tmax'])
            current_set = History.copy()
            for k in range(self.options['GenerationNum']):
                future_set = []
                for i in range(current_set.shape[1]):
                    ti = current_set[0, i]
                    ui = int(current_set[1, i])
                    print('ui:', ui)
                    t = 0

                    phi_t = ImpactFunction(ui, t, para)
                    mt = sum(phi_t)

                    while t < self.options['Tmax']-ti:

                        s = np.random.exponential(1/mt)[0]
                        U = random.rand()

                        phi_ts = ImpactFunction(ui,t+s, para)
                        mts = sum(phi_ts)

                        #print('s = %f')

                        if t+s > self.options['Tmax']-ti or U > mts/mt:
                            t = t+s
                        else:
                            u = random.random()*mts
                            sumIs =0 
                            for d in range(phi_ts):
                                sumIs += phi_ts(d)
                                if sumIs >= u:
                                    break
                            index = d

                            t += s
                            future_set.append([t + ti, index])
                        phi_t = ImpactFunction(ui, t, para)
                        mt = np.sum(phi_t)
                    future_set = np.array(future_set).T
                if len(future_set)==0 or History.shape[1] > self.options['Nmax']:
                    break
                else:
                    current_set = future_set
                    History = np.concatenate((History,current_set),axis=1)
            
            index = np.argsort(History[0, : ])
            #Seqs(n).Time = History(1,index);
            #Seqs(n).Mark = History(2,index);
            Seqs.append({
            'Time': History[0, index],
            'Mark': History[1, index],
            'Start': 0,
            'Stop': self.options['Tmax']
            })
            index = np.where(Seqs[n]['Time'] <= self.options['Tmax'])[0]
            Seqs[n]['Time'] = Seqs[n]['Time'][index]
            Seqs[n]['Mark'] = Seqs[n]['Mark'][index]
        return Seqs
    
    def simulation(self):
        pass



if __name__ == '__main__':

    K = 2 #the number of cluster
    paras = get_paras()
    options = options
    s_hp = simulatino_HP(options)
    seqs=[]
    for para in paras:
        seq = s_hp.generation(para)
        seqs.append(seq)

    #定制属于每个k的hawkes process的参数para


