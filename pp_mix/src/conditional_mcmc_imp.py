import numpy as np
from scipy.stats import gamma, multivariate_normal
from scipy.special import softmax
from .precs.precmat import PrecMat

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

        self.na_means = self.pp_mix.sample_uniform(1)
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
