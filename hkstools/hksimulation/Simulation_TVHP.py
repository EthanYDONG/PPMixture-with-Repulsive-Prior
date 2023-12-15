import numpy as np
import random
import time 
from .Intensity_TVHP import Intensity_TVHP

def Simulation_TVHP(N, T, mu, w, Period, Shift, MaxInfect, Type):
    Seqs = {'Time': [], 'Mark': [], 'Start': [], 'Stop': [], 'Feature': []}

    for ns in range(1, N + 1):
        t = 0
        History = np.array([]).reshape(2, 0)

        lambda_t = Intensity_TVHP(t, History, T, mu, w, Period, Shift, MaxInfect, Type)
        mt = np.sum(lambda_t)

        while t < T:
            s = random.expovariate(1/mt)
            U = random.random()

            lambda_ts = Intensity_TVHP(t+s, History, T, mu, w, Period, Shift, MaxInfect, Type)
            mts = np.sum(lambda_ts)

            if t + s > T or U > mts / mt:
                t = t + s
                mt = mts
            else:
                u = random.random() * mts
                sumIs = 0
                for d, value in enumerate(lambda_ts):
                    sumIs += value
                    if sumIs >= u:
                        break
                index = d

                t = t + s
                History = np.column_stack([History, [t, index]])

                lambda_t = Intensity_TVHP(t, History, T, mu, w, Period, Shift, MaxInfect, Type)
                mt = np.sum(lambda_t)

        Seqs['Time'].append(list(History[0, :]))
        Seqs['Mark'].append(list(History[1, :]))
        Seqs['Start'].append(0)
        Seqs['Stop'].append(T)

        index = [i for i, time in enumerate(Seqs['Time'][-1]) if time <= T]
        Seqs['Time'][-1] = [Seqs['Time'][-1][i] for i in index]
        Seqs['Mark'][-1] = [Seqs['Mark'][-1][i] for i in index]

        if ns % 10 == 0 or ns == N:
            print('Type={}, #seq={}/{}, #event={}, time={:.2f}sec'.format(
                Type, ns, N, len(Seqs['Mark'][-1]), time.time()))

    return Seqs

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def plot_sequences(sequences):
        for i, (times, marks) in enumerate(zip(sequences['Time'], sequences['Mark'])):
            plt.step(times, marks + i * 0.1, where='post', label=f'Sequence {i + 1}')

        plt.xlabel('Time')
        plt.ylabel('Event Type')
        plt.title('Simulated Sequences')
        plt.legend()
        plt.show()

    # 使用示例参数
    N = 5
    T = 50
    mu = np.array([0.5, 0.3])
    w = 0.1
    Period = 10
    Shift = 2
    MaxInfect = 2
    Type = 1

    # 调用模拟函数
    simulation_result = Simulation_TVHP(N, T, mu, w, Period, Shift, MaxInfect, Type)

    # 可视化
    plot_sequences(simulation_result)