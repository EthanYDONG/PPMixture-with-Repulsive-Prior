import numpy as np
from scipy.stats import uniform
from scipy.stats import poisson
from scipy.stats import truncnorm

class Point:
    def __init__(self):
        self.coords = None
        self.number = None
        self.r_mark = None

class PerfectSampler:
    def __init__(self, pp):
        self.pp = pp
        self.id2point = {}
        self.in_lower = []
        self.in_upper = []
        self.transitions = []
        self.state = []
        self.max_id = 0
        self.numupper = 0
        self.numlower = 0
        self.double_t = 2

    def initialize(self):
        self.in_upper = []
        self.in_lower = []
        # Initialize points with the dominating poisson process
        npoints = poisson.rvs(self.pp.get_cstar())
        for i in range(npoints):
            p = Point()
            p.coords = self.pp.phi_star_rng()
            p.number = self.max_id
            self.max_id += 1
            self.state.insert(0, p)
            self.id2point[p.number] = p
        #print("init npoints:", npoints)

    def estimate_doubling_time(self):
        # Run the perfect backward birth and death until all the original points
        # have been deleted
        start_npoints = len(self.state)
        t = 0
        while start_npoints > 0:
            self.one_backward(self.state)
            if self.transitions[0][1] and self.transitions[0][0] <= start_npoints:
                start_npoints -= 1
            t += 1
        self.double_t = t
        self.transitions = []

    def one_backward(self, points):
        rsec = uniform.rvs(0, 1)
        cstar = self.pp.get_cstar()

        if rsec > cstar / (cstar + len(points)):
            if len(points) == 0:
                return

            rthird = uniform.rvs(0, 1)
            probas = np.ones(len(points)) / len(points)
            removed = np.random.choice(len(points), p=probas)
            p = points.pop(removed)
            p.r_mark = rthird
            self.transitions.insert(0, (p.number, True, rthird))
        else:
            p = Point()
            p.coords = self.pp.phi_star_rng()
            p.number = self.max_id
            points.insert(0, p)
            self.max_id += 1
            self.transitions.insert(0, (p.number, False, -1.0))
            self.id2point[p.number] = p

    def one_forward(self, trans):
        curr = self.id2point[trans[0]]
        if trans[1]:
            upper = [self.id2point[id] for id in self.in_upper][::-1]
            lower = [self.id2point[id] for id in self.in_lower][::-1]

            rthird = trans[2]
            if np.log(rthird) < self.pp.papangelou_point(curr, lower) - self.pp.phi_star_dens(curr.coords):
                self.in_upper.insert(0, curr.number)
                self.numupper += 1
            if np.log(rthird) < self.pp.papangelou_point(curr, upper) - self.pp.phi_star_dens(curr.coords):
                self.in_lower.insert(0, curr.number)
                self.numlower += 1

        else:
            if len(self.in_upper) > 0:
                self.in_upper = [id for id in self.in_upper if id != curr.number]
            if len(self.in_lower) > 0:
                self.in_lower = [id for id in self.in_lower if id != curr.number]

    def simulate(self):
        self.initialize()

        # self.estimate_doubling_time()
        # print("estimated doubling time")

        if len(self.state) == 0:
            out = np.zeros((len(self.state), self.pp.get_dim()))
            for i, p in enumerate(self.state):
                out[i, :] = p.coords
            return out

        start_t = 0
        end_t = 32
        is_first = True
        coalesced = False

        iter = 0
        while not coalesced:
            iter += 1
            # print("iter:", iter)
            for i in range(start_t, end_t):
                self.one_backward(self.state)

            self.in_upper = [p.number for p in self.state][::-1]

            self.in_lower = []
            for i in range(end_t):
                trans_it = self.transitions[i]
                self.one_forward(trans_it)
                

            start_t = end_t
            end_t *= 2
            coalesced = (len(self.in_lower) == len(self.in_upper))

        out = np.zeros((len(self.in_upper), self.pp.get_dim()))
        for i, id in enumerate(self.in_upper):
            out[i, :] = self.id2point[id].coords
        return out
