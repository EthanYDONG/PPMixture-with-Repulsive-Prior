import numpy as np
from .Kernel import Kernel
def SupIntensity_HP(t, History, para, options):
    """
    计算 Hawkes 过程强度函数的上限

    参数:
    t: 当前时间
    History: 历史事件序列
    para: Hawkes 过程的参数
    options: 选项参数

    返回:
    mt: 强度函数的上限
    """
    if not History:
        mt = np.sum(para['mu'])
    else:
        Time = History[0, :]
        index = Time <= t
        Time = Time[index]
        Event = History[1, index]

        MT = np.sum(para['mu']) * np.ones(options['M'])
        for m in range(1, options['M']+1):
            t_current = t + (m - 1) * options['tstep'] / options['M']

            # 计算基函数
            basis = Kernel(t_current - Time, para)
            A = para['A'][Event, :, :]

            for c in range(0, para['A'].shape[2]):
                MT[m-1] = MT[m-1] + np.sum(basis * A[:,:,c])

        mt = np.max(MT)

    # 避免负值，确保强度非负
    mt = mt * (mt > 0)

    return mt

if __name__ == '__main__':
    print('ok')
    SupIntensity_HP(0, 0, 0, 0)
    

