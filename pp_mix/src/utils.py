import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

def delete_row(x, ind):
    if ind < x.shape[0] - 1:
        x[ind:, :] = x[ind + 1:, :]
    return x[:-1, :]

def delete_elem(x, ind):
    if ind < len(x) - 1:
        x[ind:] = x[ind + 1:]
    return x[:-1]

def vstack(rows):
    return np.vstack(rows)

def o_multi_normal_prec_lpdf(x, mu, sigma):
    neg_log_sqrt_two_pi = -0.5 * np.log(2 * np.pi)
    out = 0.5 * sigma.log_det() + neg_log_sqrt_two_pi * len(x)
    out -= 0.5 * (sigma.cho_factor_eval() @ (x - mu)).T @ (sigma.cho_factor_eval() @ (x - mu))
    return out

def o_multi_normal_prec_lpdf_vectorized(x, mu, sigma):
    neg_log_sqrt_two_pi = -0.5 * np.log(2 * np.pi)
    out = 0.5 * sigma.log_det() * x.shape[0]
    
    cho_sigma = sigma.cho_factor_eval()
    loglikes = np.sum((cho_sigma @ (x - mu))**2, axis=1)
    
    out -= np.sum(loglikes)
    return 0.5 * out

def trunc_normal_rng(mu, sigma, lower, upper):
    while True:
        val = np.random.normal(mu, sigma)
        if lower <= val <= upper:
            return val

def trunc_normal_rng_inversion(mu, sigma, lower, upper):
    u = np.random.uniform(norm.cdf((lower - mu) / sigma), norm.cdf((upper - mu) / sigma))
    tmp = norm.ppf(u)
    return sigma * tmp + mu

def trunc_normal_lpdf(x, mu, sigma, lower, upper):
    if (x < lower) or (x > upper):
        return -np.inf

    out = norm.logpdf(x, mu, sigma)
    out -= logsumexp([norm.logcdf(upper, mu, sigma), norm.logcdf(lower, mu, sigma)])

    return out

def to_proto(mat, out):
    out.rows = mat.shape[0]
    out.cols = mat.shape[1]
    out.data.extend(mat.flatten())

def to_proto_vector(vec, out):
    out.size = len(vec)
    out.data.extend(vec)

def to_eigen(vec):
    return np.array(vec.data)

def to_eigen_matrix(mat):
    return np.array(mat.data).reshape(mat.rows, mat.cols)

def to_vector_of_vectors(mat):
    return [mat[i, :] for i in range(mat.shape[0])]

def pairwise_dist_sq(x, y=None):
    if y is None:
        y = x
    return np.sum((x[:, np.newaxis, :] - y[np.newaxis, :, :])**2, axis=2)

def softmax(logs):
    num = np.exp(logs - np.max(logs))
    return num / np.sum(num)

def posterior_sim_matrix(alloc_chain):
    out = np.zeros((alloc_chain.shape[1], alloc_chain.shape[1]))
    for i in range(1, alloc_chain.shape[1]):
        for j in range(1, alloc_chain.shape[1]):
            val = np.mean(alloc_chain[:, i] == alloc_chain[:, j])
            out[i, j] = val
            out[j, i] = val
    return out


#Note that I've added a vectorized version of the o_multi_normal_prec_lpdf 
# function to efficiently handle a vector of vectors. The softmax function is 
# implemented using the np.exp and np.sum functions in NumPy. 