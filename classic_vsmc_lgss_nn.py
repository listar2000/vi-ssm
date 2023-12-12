import csv
import sys

sys.path.append('./')

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
from classic_vsmc import *


def init_neural_net_params(input_size, hidden_size, output_size, rs=npr.RandomState(0)):
    W1 = rs.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros(hidden_size)
    W2 = rs.randn(hidden_size, hidden_size) * 0.01
    b2 = np.zeros(hidden_size)
    W3 = rs.randn(hidden_size, output_size) * 0.01
    b3 = np.zeros(output_size)
    return [(W1, b1), (W2, b2), (W3, b3)]

def relu(x):
    return np.maximum(x, 0)

def neural_net_predict(params, comb_x):
    for W, b in params[:-1]:
        comb_x = np.dot(comb_x, W) + b
        comb_x = relu(comb_x)  # Applying ReLU activation
    out_W, out_b = params[-1]
    return np.dot(comb_x, out_W) + out_b

def init_model_params(Dx, Dy, alpha, r, obs, rs=npr.RandomState(0)):
    mu0 = np.zeros(Dx)
    Sigma0 = np.eye(Dx)

    A = np.zeros((Dx, Dx))
    for i in range(Dx):
        for j in range(Dx):
            A[i, j] = alpha ** (abs(i - j) + 1)

    Q = np.eye(Dx)
    C = np.zeros((Dy, Dx))
    if obs == 'sparse':
        C[:Dy, :Dy] = np.eye(Dy)
    else:
        C = rs.normal(size=(Dy, Dx))
    R = r * np.eye(Dy)

    return (mu0, Sigma0, A, Q, C, R)


def init_prop_params(T, Dx, scale=0.5, rs=npr.RandomState(0)):
    return [(scale * rs.randn(Dx),  # Bias
             1. + scale * rs.randn(Dx),  # Linear times A/mu0
             scale * rs.randn(Dx))  # Log-var
            for t in range(T)]


def generate_data(model_params, T=5, rs=npr.RandomState(0)):
    mu0, Sigma0, A, Q, C, R = model_params
    Dx = mu0.shape[0]
    Dy = R.shape[0]

    x_true = np.zeros((T, Dx))
    y_true = np.zeros((T, Dy))

    for t in range(T):
        if t > 0:
            x_true[t, :] = rs.multivariate_normal(np.dot(A, x_true[t - 1, :]), Q)
        else:
            x_true[0, :] = rs.multivariate_normal(mu0, Sigma0)
        y_true[t, :] = rs.multivariate_normal(np.dot(C, x_true[t, :]), R)

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


class lgss_smc:
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
        mean_params, sigma_params = prop_params[0], prop_params[1]
        t_column = np.full((Xp.shape[0], 1), t)
        comb_x = np.concatenate((Xp, t_column), axis=1)
        mu = neural_net_predict(mean_params, comb_x)
        s2t = np.exp(neural_net_predict(sigma_params, t)[0])
        return self.log_normal(Xc, mu, np.diag(s2t))

    def log_target(self, t, Xc, Xp, y, prop_params, model_params):
        mu0, Sigma0, A, Q, C, R = model_params
        if t > 0:
            logF = self.log_normal(Xc, np.dot(A, Xp.T).T, Q)
        else:
            logF = self.log_normal(Xc, mu0, Sigma0)
        logG = self.log_normal(np.dot(C, Xc.T).T, y[t], R)
        return logF + logG

    # These following 2 are the only ones needed by variational-smc.py
    def log_weights(self, t, Xc, Xp, y, prop_params, model_params):
        return self.log_target(t, Xc, Xp, y, prop_params, model_params) - \
               self.log_prop(t, Xc, Xp, y, prop_params, model_params)

    def sim_prop(self, t, Xp, y, prop_params, model_params, rs=npr.RandomState(0)):
        # mu0, Sigma0, A, Q, C, R = model_params
        mean_params, sigma_params = prop_params[0], prop_params[1]
        t_column = np.full((Xp.shape[0], 1), t)
        comb_x = np.concatenate((Xp, t_column), axis=1)
        mu = neural_net_predict(mean_params, comb_x)
        s2t = np.exp(neural_net_predict(sigma_params, t)[0])
        return mu + rs.randn(*Xp.shape) * np.sqrt(s2t)


def train_vsmc_lgss(T, Dx, Dy, alpha, r, N, obs, seed_n):
    # Training parameters
    param_scale = 0.5
    num_epochs = 1000
    step_size = 0.001

    data_seed = npr.RandomState(0)
    model_params = init_model_params(Dx, Dy, alpha, r, obs, data_seed)

    print("Generating data...")
    x_true, y_true = generate_data(model_params, T, data_seed)

    lml = log_marginal_likelihood(model_params, T, y_true)
    print("True log-marginal likelihood: " + str(lml))

    seed = npr.RandomState(2023)

    # Initialize proposal parameters; +1 for the time feature
    mean_params = init_neural_net_params(Dx + 1, 32, Dx)
    sigma_params = init_neural_net_params(1, 32, Dx)
    prop_params = [mean_params, sigma_params]

    lgss_smc_obj = lgss_smc(T, Dx, Dy, N)

    # Define training objective
    def objective(prop_params, iter):
        return -vsmc_lower_bound(prop_params, model_params, y_true, lgss_smc_obj, seed)


    # Get gradients of objective using autograd.
    objective_grad = grad(objective)

    print("     Epoch     |    ELBO  ")

    elbos = []
    def print_perf(prop_params, iter, grad):
        if iter % 10 == 0:
            bound = -objective(prop_params, iter)
            message = "{:15}|{:20}".format(iter, bound)
            elbos.append(bound)
            print(message)


    # SGD with adaptive step-size "adam"
    optimized_params = adam(objective_grad, prop_params, step_size=step_size,
                            num_iters=num_epochs, callback=print_perf)
    # opt_model_params, opt_prop_params = optimized_params
    return elbos


if __name__ == '__main__':
    # Model hyper-parameters
    Dxs = [5, 10, 5, 10]
    Dys = [1, 1, 3, 10]
    obss = ['sparse', 'dense', 'dense', 'sparse']
    T = 10
    alpha = 0.42
    r = .1
    N = 5

    for i in range(len(Dxs)):
        Dx, Dy, obs = Dxs[i], Dys[i], obss[i]
        name = f"data/lgss_nn_Dx_{Dx}_Dy_{Dy}.csv"
        elbo_lists = []
        for j in range(10):
            elbo_lists.append(train_vsmc_lgss(T, Dx, Dy, alpha, r, N, obs, 2020 + j))
        with open(name, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write rows to the CSV file
            for k in range(len(elbo_lists[0])):
                row = [10 * k] + [lst[k] for lst in elbo_lists]  # Replace ... with other lists
                writer.writerow(row)