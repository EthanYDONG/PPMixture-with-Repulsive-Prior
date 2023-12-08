class EigenMatrix:
    def __init__(self, data):
        self.rows = data.shape[0]
        self.cols = data.shape[0]
        self.data = data

class EigenVector:
    def __init__(self, data):
        self.size = len(data)
        self.data = data
    
    # 自定义extend函数，表示在EigenVector中增加元素
    def extend_EigenVector(self, value):
        self.data.extend(value)
        self.size += len(value)

class PPState:
    def __init__(self):
        pass

class StraussState(PPState):
    #def __init__(self, beta, gamma, R, birth_prob, birth_arate):
    def __init__(self):
        self.beta = 0
        self.gamma = 0
        self.R = 0
        self.birth_prob = []
        self.birth_arate = []

class NrepState(PPState):
    def __init__(self, u, p, tau):
        self.u = u
        self.p = p
        self.tau = tau

class MultivariateMixtureState:
    def __init__(self):
        self.ma = 1
        self.mna = 2
        self.mtot = 3
        ## cpp:
        ## repeated EigenVector a_means = 4;
        ## repeated EigenVector na_means = 5;
        #### a_means和na_means为列表，列表中每个元素是EigenVector类
        self.a_means = []
        self.na_means = []
        ## repeated EigenMatrix a_precs = 6;
        ## repeated EigenMatrix na_precs = 7;
        self.a_precs = []
        self.na_precs = []
        self.a_jumps = []
        self.na_jumps = []
        self.clus_alloc = []
        self.u = 11
        self.pp_state = PPState()
    """
    def add_EigenVector_a_means(self,value):
        a_instance = EigenVector(value)
        return self.a_means.append(a_instance)
    def add_EigenVector_na_means(self,value):
        na_instance = EigenVector(value)
        return self.na_means.append(na_instance)
    def add_EigenMatrix_a_precs(self,value):
        a_prec_instance = EigenMatrix(value)
        return self.a_precs.append(a_prec_instance)
    def add_EigenMatrix_na_precs(self,value):
        na_prec_instance = EigenMatrix(value)
        return self.na_precs.append(na_prec_instance)"""

class UnivariateMixtureState:
    def __init__(self):
        self.ma = 1
        self.mna = 2
        self.mtot = 3
        # self.a_means = EigenVector()
        # self.na_means = EigenVector()
        # self.a_precs = EigenMatrix()
        # self.na_precs = EigenMatrix()
        # self.a_jumps = EigenVector()
        # self.na_jumps = EigenVector()
        self.a_means = []
        self.na_means = 0
        self.a_precs = 0
        self.na_precs = 0
        self.a_jumps = 0
        self.na_jumps = 0
        self.clus_alloc = []
        self.u = 11
        self.pp_state = PPState()



class BernoulliMixtureState:
    def __init__(self):
        self.ma = 1
        self.mna = 2
        self.mtot = 3
        ## repeated EigenVector a_probs = 4;
        ## repeated EigenVector na_probs = 5;
        self.a_probs = []
        self.na_probs = []
        ## repeated EigenMatrix a_precs = 6;
        ## repeated EigenMatrix na_precs = 7;
        self.a_jumps = EigenVector()
        self.na_jumps = EigenVector()
        self.clus_alloc = []
        self.u = 11
        self.pp_state = PPState()
    """
    def add_EigenVector_a_probs(self,value):
        a_instance = EigenVector(value)
        return self.a_probs.append(a_instance)
    def add_EigenVector_na_probs(self,value):
        na_instance = EigenVector(value)
        return self.na_probs.append(na_instance)
    def add_EigenMatrix_a_precs(self,value):
        a_prec_instance = EigenMatrix(value)
        return self.a_precs.append(a_prec_instance)
    def add_EigenMatrix_na_precs(self,value):
        na_prec_instance = EigenMatrix(value)
        return self.na_precs.append(na_prec_instance)"""



