import csv
import sys

from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

sys.path.append('./')

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
from classic_vsmc import *


def init_model_params(Dx, mu, phi, beta, q, rs=npr.RandomState(0)):
    Q = np.diag(q)
    R = np.eye(Dx)
    mu0 = mu
    Sigma0 = Q
    return (mu0, Sigma0, mu, phi, beta, Q, R)


def init_prop_params(T, Dx, scale=0.5, rs=npr.RandomState(0)):
    return [(scale * rs.randn(Dx),  # Bias
             scale * rs.randn(Dx))  # Log-var
            for t in range(T)]


def generate_data(model_params, T=24, rs=npr.RandomState(0)):
    mu0, Sigma0, mu, phi, beta, Q, R = model_params
    Dx = mu.shape[0]
    Dy = Dx

    x_true = np.zeros((T, Dx))
    y_true = np.zeros((T, Dy))

    for t in range(T):
        if t > 0:
            x_true[t, :] = rs.multivariate_normal(mu + phi * (x_true[t-1, :] - mu), Q)
        else:
            x_true[0, :] = rs.multivariate_normal(mu0, Sigma0)
        y_true[t, :] = beta * np.exp(x_true[t, :] / 2) * rs.multivariate_normal(np.zeros(Dx), R)

    return x_true, y_true


def log_marginal_likelihood(model_params, T, y_true):
    mu0, Sigma0, A, Q, C, R = model_params
    Dx = mu0.shape[0]
    Dy = R.shape[1]

    log_likelihood = 0.
    xfilt = np.zeros(Dx)
    Pfilt = np.zeros((Dx, Dx))
    xpred = mu0
    Ppred = Sigma0

    for t in range(T):
        if t > 0:
            # Predict
            xpred = np.dot(A, xfilt)
            Ppred = np.dot(A, np.dot(Pfilt, A.T)) + Q

        # Update
        yt = y_true[t, :] - np.dot(C, xpred)
        S = np.dot(C, np.dot(Ppred, C.T)) + R
        K = np.linalg.solve(S, np.dot(C, Ppred)).T
        xfilt = xpred + np.dot(K, yt)
        Pfilt = Ppred - np.dot(K, np.dot(C, Ppred))

        sign, logdet = np.linalg.slogdet(S)
        log_likelihood += -0.5 * (np.sum(yt * np.linalg.solve(S, yt)) + logdet + Dy * np.log(2. * np.pi))

    return log_likelihood


def sample_from_product_distribution(mu_f, Sigma_f, mu_N, Sigma_N):
    # Calculate the combined mean and covariance
    Sigma_f_inv = np.linalg.inv(Sigma_f)
    Sigma_N_inv = np.linalg.inv(Sigma_N)
    Sigma_r_inv = Sigma_f_inv + Sigma_N_inv
    Sigma_r = np.linalg.inv(Sigma_r_inv)
    mu_r = Sigma_r @ (Sigma_f_inv @ mu_f + Sigma_N_inv @ mu_N)

    # Sample from the combined normal distribution
    return mu_r, Sigma_r
    # samples = np.random.multivariate_normal(mu_r, Sigma_r, size)
    #
    # return samples


class dmm_smc:
    """
    Class for defining functions used in variational SMC.
    """

    def __init__(self, T, Dx, Dy, N):
        self.T = T
        self.Dx = Dx
        self.Dy = Dy
        self.N = N

    def log_normal(self, x, mu, Sigma):
        dim = Sigma.shape[0]
        sign, logdet = np.linalg.slogdet(Sigma)
        log_norm = -0.5 * dim * np.log(2. * np.pi) - 0.5 * logdet
        Prec = np.linalg.inv(Sigma)
        return log_norm - 0.5 * np.sum((x - mu) * np.dot(Prec, (x - mu).T).T, axis=1)

    def log_prop(self, t, Xc, Xp, y, prop_params, model_params):
        mu0, Sigma0, mu, phi, beta, Q, R = model_params
        mut, log_s2t = prop_params[t]
        s2t = np.exp(log_s2t)

        if t > 0:
            mu_f = mu + phi * (Xp - mu)
            log_n = np.zeros(Xc.shape[0])
            for i in range(Xc.shape[0]):
                mu_r, Sigma_r = sample_from_product_distribution(mu_f[i,:], Q, mut, np.diag(s2t))
                if "_value" in dir(mu_r):
                    log_n[i] = multivariate_normal(mu_r._value, Sigma_r._value).logpdf(Xp[i, :])
                else:
                    log_n[i] = multivariate_normal(mu_r, Sigma_r).logpdf(Xp[i, :])
            return log_n
        else:
            return self.log_normal(Xc, mut, np.diag(s2t))

    def log_target(self, t, Xc, Xp, y, prop_params, model_params):
        mu0, Sigma0, mu, phi, beta, Q, R = model_params
        if t > 0:
            logF = self.log_normal(Xc, mu + phi * (Xp - mu), Q)
        else:
            # print(Xc.shape, mu0.shape, Sigma0.shape)
            logF = self.log_normal(Xc, mu0, Sigma0)
        logG = np.zeros(Xc.shape[0])
        for i in range(len(logG)):
            cov_mat = np.diag(beta * np.exp(Xc[i, :] / 2))
            logG[i] = multivariate_normal.pdf(y[i], mean=np.zeros(Xc.shape[1]), cov=cov_mat)
        return logF + logG

    # These following 2 are the only ones needed by variational-smc.py
    def log_weights(self, t, Xc, Xp, y, prop_params, model_params):
        return self.log_target(t, Xc, Xp, y, prop_params, model_params) - \
               self.log_prop(t, Xc, Xp, y, prop_params, model_params)

    def sim_prop(self, t, Xp, y, prop_params, model_params, rs=npr.RandomState(0)):
        mu0, Sigma0, mu, phi, beta, Q, R = model_params
        mut, log_s2t = prop_params[t]
        s2t = np.exp(log_s2t)

        if t > 0:
            mu_f = mu + phi * (Xp - mu)
            samps = np.zeros(Xp.shape)
            for i in range(Xp.shape[0]):
                mu_r, Sigma_r = sample_from_product_distribution(mu_f[i, :], Q, mut, np.diag(s2t))
                if "_value" in dir(mu_r):
                    samp = rs.multivariate_normal(mu_r._value, Sigma_r._value)
                else:
                    samp = rs.multivariate_normal(mu_r, Sigma_r)
                samps[i, :] = samp
            return samps
            # mu_r, Sigma_r = sample_from_product_distribution(mu_f, Q, mut, np.diag(s2t))
            # return rs.multivariate_normal(mu_r, Sigma_r)
        else:
            init_x = rs.multivariate_normal(mu0, Sigma0, size=Xp.shape[0])
            return init_x


def train_vsmc_lgss(Dx, T, N, seed_n):
    # Training parameters
    param_scale = 0.5
    num_epochs = 1500
    step_size = 0.001

    Dy = Dx
    data_seed = npr.RandomState(0)
    model_params = init_model_params(Dx, np.ones(Dx), 0.5 * np.ones(Dx), 0.5, 0.1 * np.ones(Dx), data_seed)

    print("Generating data...")
    x_true, y_true = generate_data(model_params, T, data_seed)
    # plt.plot(y_true)
    # plt.show()
    # assert False

    # lml = log_marginal_likelihood(model_params, T, y_true)
    # print("True log-marginal likelihood: " + str(lml))

    seed = npr.RandomState(seed_n)

    # Initialize proposal parameters
    prop_params = init_prop_params(T, Dx, param_scale, seed)

    lgss_smc_obj = dmm_smc(T, Dx, Dy, N)


    # Define training objective
    def objective(prop_params, iter):
        return -vsmc_lower_bound(prop_params, model_params, y_true, lgss_smc_obj, seed)


    # Get gradients of objective using autograd.
    objective_grad = grad(objective)

    print("     Epoch     |    ELBO  ")
    # f_head = './dmm'
    # with open(f_head + '_ELBO.csv', 'w') as f_handle:
    #     f_handle.write("iter,ELBO\n")

    elbos = []
    def print_perf(prop_params, iter, grad):
        if iter % 10 == 0:
            bound = -objective(prop_params, iter)
            message = "{:15}|{:20}".format(iter, bound)

            # with open(f_head + '_ELBO.csv', 'a') as f_handle:
            #     np.savetxt(f_handle, [[iter, bound]], fmt='%i,%f')
            elbos.append(bound)

            print(message)
    # SGD with adaptive step-size "adam"
    optimized_params = adam(objective_grad, prop_params, step_size=step_size,
                            num_iters=num_epochs, callback=print_perf)
    # opt_model_params, opt_prop_params = optimized_params
    return elbos

if __name__ == '__main__':
    # Model hyper-parameters
    Dxs = [5, 10]
    T = 24
    N = 5

    for i in range(len(Dxs)):
        Dx = Dxs[i]
        name = f"data/dmm_Dx_{Dx}.csv"
        elbo_lists = []
        for j in range(5):
            elbo_lists.append(train_vsmc_lgss(Dx, T, N, 2020 + j))
        with open(name, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write rows to the CSV file
            for k in range(len(elbo_lists[0])):
                row = [10 * k] + [lst[k] for lst in elbo_lists]  # Replace ... with other lists
                writer.writerow(row)