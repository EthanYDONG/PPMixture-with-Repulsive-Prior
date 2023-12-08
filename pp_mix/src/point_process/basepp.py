#  11-23:可能需要将basePP写为abstract base class

#from abc import ABC, abstractmethod
import numpy as np

class BasePP():
    def __init__(self):
        pass

    def set_ranges(self, ranges):
        self.ranges = ranges
        self.dim = ranges.shape[1]
        self.diff_range = (ranges[1, :] - ranges[0, :]).reshape(-1, 1)
        self.vol_range = np.prod(self.diff_range)
        self.initialize()

    def sample_uniform(self, npoints):
        out = np.zeros((npoints, self.dim))
        for j in range(self.dim):
            out[:, j] = np.random.uniform(self.ranges[0, j], self.ranges[1, j], npoints)
        return out

    def sample_given_active(self, active, non_active, psi_u):
        #print('sample_given_active begin')
        npoints = non_active.shape[0]
        c_star_na = self.c_star * psi_u
        birth_prob = np.log(c_star_na) - np.log(c_star_na + npoints)

        rsecond = np.random.uniform(0, 1)
        birth_arate = -1

        if np.log(rsecond) < birth_prob:
            # BIRTH MOVE
            xi = self.phi_star_rng()
            aux = np.vstack([active, non_active])

            #xi = xi.reshape(-1, self.dim)
            #aux = aux.reshape(-1, self.dim)
            #print(type(self),'----',xi.shape)

            pap = self.papangelou(xi, aux)                   ####################
            birth_arate = pap - self.phi_star_dens(xi) + np.log(psi_u)

            rthird = np.random.uniform(0, 1)
            if np.log(rthird) < birth_arate:
                non_active = np.vstack([non_active, xi])

        else:
            # Death Move
            if npoints == 0:
                return non_active

            probas = np.ones(npoints) / npoints
            ind = np.random.choice(np.arange(npoints), p=probas)

            non_active = np.delete(non_active, ind, axis=0)
        #print('sample_given_active end')
        return non_active
        
    def sample_n_points(self, npoints):
        max_steps = int(1e6)
        out = np.zeros((npoints, self.dim))
        logM = np.log(self.rejection_sampling_M(npoints))

        for _ in range(max_steps):
            dens_q = 0
            for k in range(npoints):
                out[k, :] = self.phi_star_rng()
                dens_q += self.phi_star_dens(out[k, :], True)

            arate = self.dens(out) - (logM + dens_q)
            u = np.random.uniform(0.0, 1.0)

            if np.log(u) < arate:
                return out

        print("MAXIMUM NUMBER OF ITERATIONS REACHED IN BasePP::sample_n_points, returning the last value")
        return out
    
    def get_ranges(self):
        return self.ranges
    
    def get_vol_ranges(self):
        return self.vol_range
    
    def get_dim(self):
        return self.ranges.shape[1]
    
    def get_cstar(self):
        return self.c_star
