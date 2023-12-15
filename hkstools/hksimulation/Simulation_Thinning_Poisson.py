import numpy as np



def Simulation_Thinning_Poisson(mu, t_start, t_end):
    """
    Implement thinning method to simulate homogeneous Poisson processes

    """

    t = t_start
    history = []
    # mu=np.exp(mu)
    mt = np.sum(mu)#mt = np.sum(mu)

    while t < t_end:
        # 生成指数分布的等待时间
        s = np.random.exponential(1/mt)
        t = t + s

        # 生成均匀分布的随机数
        u = np.random.uniform(0, mt)
        sum_is = 0

        # 根据强度的权重选择事件类型
        for d in range(len(mu)):
            sum_is = sum_is + mu[d]
            if sum_is >= u:
                break
        index = d
        
        # 记录事件的时间和类型
        history.append([t, index])

    # 筛选出在规定时间范围内的事件
    history = np.array(history).T
    index = np.where(history[0, :] < t_end)
    history = history[:, index]
    

    return history