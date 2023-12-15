import numpy as np
from scipy.stats import gamma, multivariate_normal
# from scipy.special import softmax
from .precs.precmat import PrecMat
def softmax(x):
    """计算数组 x 的 softmax。"""
    e_x = np.exp(x/1e4)  # 减去最大值来提高数值稳定性
    return e_x / e_x.sum(axis=0)

class ConditionalMCMC:
    def __init__(self, pp_mix, h, g, params):
        self.pp_mix = pp_mix
        self.h = h
        self.g = g
        self.params = params
        self.verbose = False
        self.acc_mean = 0
        self.tot_mean = 0

    def initialize(self, data):
        #
        self.data = data
        #self.ndata = data.shape[0]*data.shape[1]
        if isinstance(self.data, list):
            self.ndata = len(data)
            self.dim = np.unique(np.concatenate([entry['Mark'] for entry in data])).shape[0]

        else:
            self.ndata = data.shape[0]
        #self.dim = data[0]
            self.dim = data.shape[1]

        ranges = self.pp_mix.get_ranges()

        self.initialize_allocated_means()

        self.nclus = self.a_means.shape[0]
        self.clus_alloc = np.zeros(self.ndata, dtype=int)
        probas = np.ones(self.nclus) / self.nclus
        for i in range(self.ndata):
            self.clus_alloc[i] = np.random.choice(np.arange(self.nclus), p=probas)

        self.a_jumps = np.ones(self.nclus) / (self.nclus + 1)

        self.a_precs = [self.g.mean() for _ in range(self.nclus)]

        self.na_means = self.pp_mix.sample_uniform(1) # min(min) max(max)
        # 在 event type 
        self.na_jumps = np.ones(self.na_means.shape[0]) / (self.nclus + self.na_means.shape[0])

        self.na_precs = [self.g.mean() for _ in range(self.na_means.shape[0])]
        self.u = 1.0

        print("initialize done")

    def run_one(self):
        # get_prec_vectorized = np.vectorize(lambda x: x.get_prec() if isinstance(x, PrecMat) else x, otypes=[np.ndarray])
        # self.a_precs = get_prec_vectorized(self.a_precs)
        for i in range(len(self.a_precs)):
            self.a_precs[i] = self.a_precs[i].get_prec() if isinstance(self.a_precs[i], PrecMat) else self.a_precs[i]
        for i in range(len(self.na_precs)):
            self.na_precs[i] = self.na_precs[i].get_prec() if isinstance(self.na_precs[i], PrecMat) else self.na_precs[i]
        T = np.sum(self.a_jumps) + np.sum(self.na_jumps)
        self.u = gamma.rvs(self.ndata, scale=T)
        
        psi_u = self.h.laplace(self.u)

        self.sample_allocations_and_relabel()

        for i in range(10):
            npoints = self.na_means.shape[0]
            self.pp_mix.sample_given_active(self.a_means, self.na_means, psi_u)
            if self.na_means.shape[0] != npoints:
                break

        self.na_precs = [self.g.sample_prior() for _ in range(self.na_means.shape[0])]
        ########self.na_jumps = np.ones(self.na_means.shape[0]) / (self.nclus + self.na_means.shape[0])
        self.na_jumps = [self.h.sample_tilted(self.u) for _ in range(self.na_means.shape[0])]
        
        self.sample_means()
        self.sample_vars()
        self.sample_jumps()

        self.pp_mix.update_hypers(self.a_means, self.na_means)

    def sample_allocations_and_relabel(self):
        Ma = self.a_means.shape[0]
        Mna = self.na_means.shape[0]
        Mtot = Ma + Mna

        curr_a_means = self.a_means
        curr_na_means = self.na_means
        curr_a_precs = self.a_precs
        curr_na_precs = self.na_precs
        curr_a_jumps = self.a_jumps
        curr_na_jumps = self.na_jumps

        for i in range(self.ndata):
            probas = np.zeros(Mtot)
            newalloc = 0
            datum = self.data[i]
            probas[:Ma] = curr_a_jumps
            probas[Ma:] = curr_na_jumps
            probas = np.log(probas)

            for k in range(Ma):
                probas[k] += self.lpdf_given_clus(datum, curr_a_means[k, :], curr_a_precs[k])
                # replace with transformer
            for k in range(Mna):
                probas[k + Ma] += self.lpdf_given_clus(datum, curr_na_means[k, :], curr_na_precs[k])
                # replace with transformer

            probas = softmax(probas)
            #newalloc = np.random.choice(np.arange(Mtot), p=probas) - 1
            newalloc = np.random.choice(np.arange(Mtot), p=probas)
            self.clus_alloc[i] = newalloc

        self._relabel()

    def _relabel(self):
        na2a = set()
        a2na = set()

        Ma = self.a_means.shape[0]
        Mna = self.na_means.shape[0]
        Mtot = Ma + Mna

        for i in range(self.ndata):
            if self.clus_alloc[i] >= Ma:
                na2a.add(self.clus_alloc[i] - Ma)

        for k in range(Ma):
            if (self.clus_alloc == k).sum() == 0:
                a2na.add(k)

        a2na_vec = list(a2na)
        n_new_na = len(a2na)
        new_na_means = self.a_means[a2na_vec, :]
        new_na_precs = [self.a_precs[i] for i in a2na_vec]
        #new_na_jumps = self.a_jumps[a2na_vec]
        new_na_jumps = [self.a_jumps[i] for i in a2na_vec]

        na2a_vec = list(na2a)
        n_new_a = len(na2a)
        new_a_means = self.na_means[na2a_vec, :]
        new_a_precs = [self.na_precs[i] for i in na2a_vec]
        #new_a_jumps = self.na_jumps[na2a_vec]
        new_a_jumps = [self.na_jumps[i] for i in na2a_vec]

        for it in reversed(a2na_vec):
            self.a_means = np.delete(self.a_means, it, axis=0)
            self.a_jumps = np.delete(self.a_jumps, it)
            del self.a_precs[it]

        if self.na_means.shape[0] > 0:
            for it in reversed(na2a_vec):
                self.na_means = np.delete(self.na_means, it, axis=0)
                self.na_jumps = np.delete(self.na_jumps, it)
                del self.na_precs[it]

        if new_a_means.shape[0] > 0:
            oldMa = self.a_means.shape[0]
            self.a_means = np.concatenate((self.a_means, new_a_means), axis=0)
            self.a_jumps = np.concatenate((self.a_jumps, new_a_jumps))
            self.a_precs.extend(new_a_precs)

        if new_na_means.shape[0] > 0:
            oldMna = self.na_means.shape[0]
            self.na_means = np.concatenate((self.na_means, new_na_means), axis=0)
            self.na_jumps = np.concatenate((self.na_jumps, new_na_jumps))
            self.na_precs.extend(new_na_precs)

        old2new = {value: key for key, value in enumerate(np.unique(self.clus_alloc))}
        self.clus_alloc = np.array([old2new[i] for i in self.clus_alloc])

        self.data_by_clus = [[] for _ in range(self.a_means.shape[0])]
        for i in range(self.ndata):
            self.data_by_clus[self.clus_alloc[i]].append(self.data[i])

    def sample_means(self):
        #print('sample_means begin')
        allmeans = np.concatenate((self.a_means, self.na_means), axis=0)

        for i in range(self.a_means.shape[0]):
            self.tot_mean += 1
            others = np.delete(allmeans, i, axis=0)
            sigma = self.max_proposal_sigma if np.random.rand() < 0.1 else self.min_proposal_sigma

            currmean = self.a_means[i, :]
            cov_prop = np.eye(self.dim) * sigma
            prop = multivariate_normal.rvs(currmean, cov_prop)

            prop = np.array(prop).reshape(-1, self.dim)
            currmean = np.array(currmean).reshape(-1, self.dim)
           
            currlik = self.lpdf_given_clus_multi(self.data_by_clus[i], currmean, self.a_precs[i])
            proplik = self.lpdf_given_clus_multi(self.data_by_clus[i], prop, self.a_precs[i])
            #1.  replace lpdf_given_clus_multi with transformer
            lik_ratio = proplik - currlik
            prior_ratio = self.pp_mix.papangelou(prop, others) - self.pp_mix.papangelou(currmean, others)

            arate = lik_ratio + prior_ratio

            accepted = False
            if np.log(np.random.rand()) < arate:
                accepted = True
                self.a_means[i, :] = prop
                self.acc_mean += 1

            if self.verbose:
                print("Component:", i)
                print("data:")
                self.print_data_by_clus(i)
                print("currmean:", currmean, ", currlik:", currlik)
                print("prop:", prop, ", proplik:", proplik)
                print("prior_ratio:", prior_ratio)
                print("prop_papangelou:", self.pp_mix.papangelou(prop, others), ", curr_papangelou:",
                      self.pp_mix.papangelou(currmean, others))
                print("lik_ratio:", lik_ratio)
                print("ACCEPTED:", accepted)
                print("**********")
        # print('sample_means end')
    def sample_vars(self):
        
        for i in range(self.a_means.shape[0]):##########################
            self.a_precs[i] = self.g.sample_given_data(self.data_by_clus[i], self.a_precs[i],
                                                       self.a_means[i, :])
        #### do with 
        # for i in range(self.a_means.shape[0]):
        #     self.tot_mean += 1
        #     others = np.delete(allmeans, i, axis=0)
        #     sigma = self.max_proposal_sigma if np.random.rand() < 0.1 else self.min_proposal_sigma

        #     currmean = self.a_precs[i, :]
        #     cov_prop = np.eye(self.dim) * sigma
        #     prop = multivariate_normal.rvs(currmean, cov_prop)

        #     prop = np.array(prop).reshape(-1, self.dim)
        #     currmean = np.array(currmean).reshape(-1, self.dim)
           
        #     currlik = self.lpdf_given_clus_multi(self.data_by_clus[i], currmean, self.a_precs[i])
        #     proplik = self.lpdf_given_clus_multi(self.data_by_clus[i], prop, self.a_precs[i])
        #     #1.  replace lpdf_given_clus_multi with transformer
        #     lik_ratio = proplik - currlik
        #     #prior_ratio = self.pp_mix.papangelou(prop, others) - self.pp_mix.papangelou(currmean, others)
        #     prior_ration = self.pp_mix.lpdf(prop) - self.pp_mix.lpdf(currmean)
        #     arate = lik_ratio + prior_ratio

        #     accepted = False
        #     if np.log(np.random.rand()) < arate:
        #         accepted = True
        #         self.a_means[i, :] = prop
        #         self.acc_mean += 1



    def sample_jumps(self):
        for i in range(self.a_means.shape[0]):
            self.a_jumps[i] = self.h.sample_given_data(len(self.data_by_clus[i]), self.a_jumps[i], self.u)

    def print_debug_string(self):
        print("*********** DEBUG STRING***********")
        print("#### ACTIVE: Number actives:", self.a_means.shape[0])
        for i in range(self.a_means.shape[0]):
            print("Component:", i, ", weight:", self.a_jumps[i], ", mean:", self.a_means[i, :])

        print("#### NON - ACTIVE: Number actives:", self.na_means.shape[0])
        for i in range(self.na_means.shape[0]):
            print("Component:", i, ", weight:", self.na_jumps[i], ", mean:", self.na_means[i, :])
        print()





class ConditionalMCMC_model: #大框架
    def __init__(self, pp_mix, h, g, params):
        self.pp_mix = pp_mix
        self.h = h
        self.g = g
        self.params = params
        self.verbose = False
        self.acc_mean = 0
        self.tot_mean = 0

    def initialize(self, data, hakes_model):
        # data = seqmix list of dict
        self.data = data
        #self.ndata = data.shape[0]*data.shape[1]
        if isinstance(self.data, list):
            self.ndata = len(data)
            self.dim = np.unique(np.concatenate([entry['Mark'] for entry in data])).shape[0]
        else:
            self.ndata = data.shape[0]
        #self.dim = data[0]
            self.dim = data.shape[1]

        # ranges = self.pp_mix.get_ranges()
        # hakes.model.ranges Event_type * 2
        self.initialize_allocated_means(hakes_model.ranges)

        hakes_model.b_a =  self.a_means.T #   self.a_means shape = k * event_type 每一行表示 一个envent_type 的 k个采样值

        self.nclus = self.a_means.shape[0]
        hakes_model.K = self.nclus
        hakes_model.cluster_num = self.nclus

        self.clus_alloc = np.zeros(self.ndata, dtype=int)
        probas = np.ones(self.nclus) / self.nclus
        for i in range(self.ndata):
            self.clus_alloc[i] = np.random.choice(np.arange(self.nclus), p=probas)

        self.a_jumps = np.ones(self.nclus) / (self.nclus + 1)
        hakes_model.R_a = self.a_jumps

        self.a_precs = [self.g.mean() for _ in range(self.nclus)]
        for k in range(self.nclus):
            # print(self.a_precs[k].shape,'--')
            hakes_model.beta_a[:,:,k,:] = self.a_precs[k]
            # print(hakes_model.beta_a.shape,'---')
        
        # self.na_means = self.pp_mix.sample_uniform(1)
        min_left = np.min(hakes_model.ranges[:, 0])
        max_right = np.max(hakes_model.ranges[:, 1])
        # self.a_means k * event_type 
        self.na_means = np.random.uniform(min_left,max_right,size=self.a_means.shape[1]).reshape(1,-1)
        hakes_model.b_na = self.na_means.T

        self.na_jumps = np.ones(self.na_means.shape[0]) / (self.nclus + self.na_means.shape[0])
        hakes_model.R_na = self.na_jumps

        self.na_precs = [self.g.mean() for _ in range(self.na_means.shape[0])]
        for i in range(self.na_means.shape[0]):
            hakes_model.beta_na[:,:,i,:] = self.na_precs[i]
        
        self.u = 1.0
        # hakes_model 初始化
        self.hakes_model = hakes_model
        # print(self.a_means.shape,hakes_model.b_a.shape)
        # print(self.na_means.shape,hakes_model.b_na.shape)
        print("initialize done")
    def model_init(self):
        pass
    def run_one(self):
        # get_prec_vectorized = np.vectorize(lambda x: x.get_prec() if isinstance(x, PrecMat) else x, otypes=[np.ndarray])
        # self.a_precs = get_prec_vectorized(self.a_precs)
        # for i in range(len(self.a_precs)):
        #     self.a_precs[i] = self.a_precs[i].get_prec() if isinstance(self.a_precs[i], PrecMat) else self.a_precs[i]
        # for i in range(len(self.na_precs)):
        #     self.na_precs[i] = self.na_precs[i].get_prec() if isinstance(self.na_precs[i], PrecMat) else self.na_precs[i]
        T = np.sum(self.a_jumps) + np.sum(self.na_jumps)
        self.u = gamma.rvs(self.ndata, scale=T)
        
        psi_u = self.h.laplace(self.u)

        self.sample_allocations_and_relabel()

        for i in range(10):
            npoints = self.na_means.shape[0]
            self.pp_mix.sample_given_active(self.a_means, self.na_means, psi_u)
            if self.na_means.shape[0] != npoints:
                break

        self.na_precs = [self.g.sample_prior() for _ in range(self.na_means.shape[0])]
        for i in range(self.na_means.shape[0]):
            if i >= self.hakes_model.beta_na.shape[2]:
                self.hakes_model.beta_na=np.append(self.hakes_model.beta_a,np.expand_dims(self.na_precs[i],axis=2),axis=2)
            else:
                self.hakes_model.beta_na[:,:,i,:] = self.na_precs[i]
        ########self.na_jumps = np.ones(self.na_means.shape[0]) / (self.nclus + self.na_means.shape[0])
        self.na_jumps = [self.h.sample_tilted(self.u) for _ in range(self.na_means.shape[0])]
        self.hakes_model.R_na = self.na_jumps

        self.sample_means()
        self.sample_vars()
        self.sample_jumps()

        self.pp_mix.update_hypers(self.a_means, self.na_means)

    def sample_allocations_and_relabel(self):
        Ma = self.a_means.shape[0]
        Mna = self.na_means.shape[0]
        Mtot = Ma + Mna

        curr_a_means = self.a_means
        curr_na_means = self.na_means
        curr_a_precs = self.a_precs
        curr_na_precs = self.na_precs
        curr_a_jumps = self.a_jumps
        curr_na_jumps = self.na_jumps

        for i in range(self.ndata):
            probas = np.zeros(Mtot)
            newalloc = 0
            datum = self.data[i]
            probas[:Ma] = curr_a_jumps
            probas[Ma:] = curr_na_jumps
            probas = np.log(probas)

            for k in range(Ma):
                # probas[k] += self.lpdf_given_clus(datum, curr_a_means[k, :], curr_a_precs[k])
                probas[k] += self.hakes_model.loglike_one_a(datum,k)
                # replace with transformer
            for k in range(Mna):
                # probas[k + Ma] += self.lpdf_given_clus(datum, curr_na_means[k, :], curr_na_precs[k])
                # print(k)
                probas[k+Ma] += self.hakes_model.loglike_one_na(datum,k)
                # replace with transformer
            #
            probas = softmax(probas)
            #newalloc = np.random.choice(np.arange(Mtot), p=probas) - 1
            # print('probas',probas)
            # import pdb
            # pdb.set_trace()
            newalloc = np.random.choice(np.arange(Mtot), p=probas)
            self.clus_alloc[i] = newalloc

        # print(self.a_means.shape)
        self._relabel()
        # print(self.a_means.shape)

    def _relabel(self):
        na2a = set()
        a2na = set()

        Ma = self.a_means.shape[0]
        Mna = self.na_means.shape[0]
        Mtot = Ma + Mna

        for i in range(self.ndata):
            if self.clus_alloc[i] >= Ma:
                na2a.add(self.clus_alloc[i] - Ma)

        for k in range(Ma):
            if (self.clus_alloc == k).sum() == 0:
                a2na.add(k)

        a2na_vec = list(a2na)
        n_new_na = len(a2na)
        new_na_means = self.a_means[a2na_vec, :]
        new_na_precs = [self.a_precs[i] for i in a2na_vec]
        #new_na_jumps = self.a_jumps[a2na_vec]
        new_na_jumps = [self.a_jumps[i] for i in a2na_vec]

        na2a_vec = list(na2a)
        n_new_a = len(na2a)
        new_a_means = self.na_means[na2a_vec, :]
        new_a_precs = [self.na_precs[i] for i in na2a_vec]
        #new_a_jumps = self.na_jumps[na2a_vec]
        new_a_jumps = [self.na_jumps[i] for i in na2a_vec]

        for it in reversed(a2na_vec):
            self.a_means = np.delete(self.a_means, it, axis=0)
            self.hakes_model.b_a = self.a_means.T
            self.a_jumps = np.delete(self.a_jumps, it)
            self.hakes_model.R_a = self.a_jumps
            self.hakes_model.K -= 1
            self.hakes_model.cluster_num -= 1
            del self.a_precs[it]
            self.hakes_model.beta_a = np.delete(self.hakes_model.beta_a,it,axis=2)

        if self.na_means.shape[0] > 0:
            for it in reversed(na2a_vec):
                self.na_means = np.delete(self.na_means, it, axis=0)
                self.hakes_model.b_na = self.na_means.T
                self.na_jumps = np.delete(self.na_jumps, it)
                self.hakes_model.R_na = self.na_jumps
                del self.na_precs[it]
                self.hakes_model.beta_na = np.delete(self.hakes_model.beta_na,it,axis=2)

        if new_a_means.shape[0] > 0:
            oldMa = self.a_means.shape[0]
            self.a_means = np.concatenate((self.a_means, new_a_means), axis=0)
            self.hakes_model.b_a = self.a_means.T
            self.a_jumps = np.concatenate((self.a_jumps, new_a_jumps))
            self.hakes_model.R_a = self.a_jumps
            self.a_precs.extend(new_a_precs)
            self.hakes_model.K += new_a_means.shape[0]
            self.hakes_model.cluster_num += new_a_means.shape[0]
            # print(len(new_a_precs))
            # print(new_a_precs[0].shape)
            # print(self.hakes_model.beta_a.shape)
            combind = np.concatenate([np.expand_dims(x,axis=2) for x in new_a_precs])

            self.hakes_model.beta_a = np.append(self.hakes_model.beta_a, combind,axis=2)

        if new_na_means.shape[0] > 0:
            oldMna = self.na_means.shape[0]
            self.na_means = np.concatenate((self.na_means, new_na_means), axis=0)
            self.hakes_model.b_na = self.na_means.T
            self.na_jumps = np.concatenate((self.na_jumps, new_na_jumps))
            self.hakes_model.R_na = self.na_jumps
            self.na_precs.extend(new_na_precs)
            ombind = np.concatenate([np.expand_dims(x,axis=2) for x in new_na_precs])
            self.hakes_model.beta_na = np.append(self.hakes_model.beta_na, combind,axis=2)

        old2new = {value: key for key, value in enumerate(np.unique(self.clus_alloc))}
        self.clus_alloc = np.array([old2new[i] for i in self.clus_alloc])
        self.hakes_model.label = self.clus_alloc.reshape(1,-1)

        self.data_by_clus = [[] for _ in range(self.a_means.shape[0])]
        for i in range(self.ndata):
            self.data_by_clus[self.clus_alloc[i]].append(self.data[i])

    def sample_means(self): # sample base intensity 
        #print('sample_means begin')
        allmeans = np.concatenate((self.a_means, self.na_means), axis=0)

        for i in range(self.a_means.shape[0]):
            self.tot_mean += 1
            others = np.delete(allmeans, i, axis=0)
            sigma = self.max_proposal_sigma if np.random.rand() < 0.1 else self.min_proposal_sigma

            currmean = self.a_means[i, :]
            cov_prop = np.eye(self.dim) * sigma
            prop = multivariate_normal.rvs(currmean, cov_prop)
            prop[prop<0]=0.01

            prop = np.array(prop).reshape(-1, self.dim)
            currmean = np.array(currmean).reshape(-1, self.dim)
            # print(self.dim,'--checkpoint')
            # currlik = self.lpdf_given_clus_multi(self.data_by_clus[i], currmean, self.a_precs[i])
            # proplik = self.lpdf_given_clus_multi(self.data_by_clus[i], prop, self.a_precs[i])
            #1.  replace lpdf_given_clus_multi with transformer
            #print(self.a_means.shape)
            #print(self.hakes_model.b_a.shape)
            currlik = self.hakes_model.Loglike(i)
            proplik = self.hakes_model.Loglike(i, mu_prop = prop.T)
            lik_ratio = proplik - currlik
            # print(prop.shape)
            # print(currmean.shape)
            # print(others.shape)
            prior_ratio = self.pp_mix.papangelou(prop, others) - self.pp_mix.papangelou(currmean, others)

            arate = lik_ratio + prior_ratio

            accepted = False
            if np.log(np.random.rand()) < arate:
                accepted = True
                self.a_means[i, :] = prop # k * event_type 
                self.hakes_model.b_a[:,i] = prop 
                self.acc_mean += 1

            if self.verbose:
                print("Component:", i)
                print("data:")
                self.print_data_by_clus(i)
                print("currmean:", currmean, ", currlik:", currlik)
                print("prop:", prop, ", proplik:", proplik)
                print("prior_ratio:", prior_ratio)
                print("prop_papangelou:", self.pp_mix.papangelou(prop, others), ", curr_papangelou:",
                      self.pp_mix.papangelou(currmean, others))
                print("lik_ratio:", lik_ratio)
                print("ACCEPTED:", accepted)
                print("**********")
        # print('sample_means end')
    def sample_vars(self): # sample A
        
        # for i in range(self.a_means.shape[0]):##########################
        #     self.a_precs[i] = self.g.sample_given_data(self.data_by_clus[i], self.a_precs[i],
        #                                                self.a_means[i, :])
        #### do with 
        for i in range(self.a_means.shape[0]):
            self.tot_mean += 1
            sigma = self.max_proposal_sigma if np.random.rand() < 0.1 else self.min_proposal_sigma
            curr_precs = self.a_precs[i]
            curr_mean = self.a_means[i, :]
            D,M,D = curr_precs.shape
            cov_prop = np.eye(D*M*D) * sigma
            # print(curr_precs.shape)
            shape = curr_precs.shape
            prop = multivariate_normal.rvs(curr_precs.flatten(), cov_prop)
            prop[prop<0]=0

            prop = np.array(prop).reshape(shape)
           
            # currlik = self.lpdf_given_clus_multi(self.data_by_clus[i], currmean, self.a_precs[i])
            # proplik = self.lpdf_given_clus_multi(self.data_by_clus[i], prop, self.a_precs[i])
            currlik = self.hakes_model.Loglike(i)
            proplik = self.hakes_model.Loglike(i, A_prop = prop)
            #1.  replace lpdf_given_clus_multi with transformer
            lik_ratio = proplik - currlik
            #prior_ratio = self.pp_mix.papangelou(prop, others) - self.pp_mix.papangelou(currmean, others)
            prior_ration = self.g.lpdf(prop) - self.g.lpdf(curr_precs)
            arate = lik_ratio + prior_ration

            accepted = False
            if np.log(np.random.rand()) < arate:
                print('A update')
                accepted = True
                self.a_precs[i] = prop
                self.hakes_model.beta_a[:,:,i,:] = prop
                self.acc_mean += 1



    def sample_jumps(self):
        for i in range(self.a_means.shape[0]):
            self.a_jumps[i] = self.h.sample_given_data(len(self.data_by_clus[i]), self.a_jumps[i], self.u)
        self.hakes_model.R_na = self.a_jumps
    def print_debug_string(self):
        print("*********** DEBUG STRING***********")
        print("#### ACTIVE: Number actives:", self.a_means.shape[0])
        for i in range(self.a_means.shape[0]):
            print("Component:", i, ", weight:", self.a_jumps[i], ", mean:", self.a_means[i, :])

        print("#### NON - ACTIVE: Number actives:", self.na_means.shape[0])
        for i in range(self.na_means.shape[0]):
            print("Component:", i, ", weight:", self.na_jumps[i], ", mean:", self.na_means[i, :])
        print()
