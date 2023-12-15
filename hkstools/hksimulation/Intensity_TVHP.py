import numpy as np
from .Infectivity_TVHP import Infectivity_TVHP

def Intensity_TVHP(t, History, T, mu, w, Period, Shift, MaxInfect, Type):
    """
    Calculate the intensity function of a time-varying multi-dimensional Hawkes process.
    
    Parameters:
    - t: Current time interval
    - History: Historical event sequence
    - T: Time interval
    - mu: U*1 vector (baseline intensity)
    - w: Decay function parameter
    - Period, Shift: Period and shift parameters of the infectivity matrix
    - MaxInfect: Maximum infectivity
    - Type: Infectivity mode
    
    Returns:
    - lambda_: Intensity
    """
    if History.size == 0:
        return mu  # 如果历史事件序列为空，直接将无条件强度作为强度函数

    Time = History[0, :]  # 提取事件发生的时间
    index = Time <= t  # 筛选在当前时间之前的事件记录
    Time = Time[index]
    Event = History[1, index]  # 提取事件类型

    lambda_ = mu  # 初始化强度函数为无条件强度

    At = Infectivity_TVHP(T, t, Period, Shift, MaxInfect, Type)  # 计算当前时间的传染性矩阵
    
    for i in range(len(Time)):
        ui = Event[i]
        
        lambda_ = lambda_ + At[:, ui] * np.exp(-w * (t - Time[i]))  # 更新强度函数，将传染性矩阵中对应事件类型的列与衰减函数的乘积加到强度函数中

    return lambda_

# 示例用法：
# t_current = 2.0
# event_current = 1
# t_old = 1.0
# lambdat = np.array([0.5, 0.6, 0.7]).reshape(-1, 1)
# para = {'mu': np.array([0.2, 0.3, 0.4]).reshape(-1, 1),
#         'A': np.random.rand(3, 1, 3),
#         'kernel': 'exp',
#         'w': 0.1}
# result = Intensity_TVHP(t_current, event_current, t_old, lambdat, para)
# print(result)