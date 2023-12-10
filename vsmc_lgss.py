import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.optim import Adam
import torch.distributions as dist
from vsmc import *


def init_model_params(Dx, Dy, alpha, r, obs, device='cpu'):
    mu0 = torch.zeros(Dx, device=device)
    Sigma0 = torch.eye(Dx, device=device)

    A = torch.zeros((Dx, Dx), device=device)
    for i in range(Dx):
        for j in range(Dx):
            A[i, j] = alpha ** (abs(i - j) + 1)

    Q = torch.eye(Dx, device=device)
    C = torch.zeros((Dy, Dx), device=device)
    if obs == 'sparse':
        C[:Dy, :Dy] = torch.eye(Dy, device=device)
    else:
        C = torch.randn((Dy, Dx), device=device)
    R = r * torch.eye(Dy, device=device)

    return mu0, Sigma0, A, Q, C, R


def init_prop_params(T, Dx, scale=0.5, device='cpu'):
    params = []
    for t in range(T):
        bias = torch.randn(Dx, device=device) * scale
        linear = torch.randn(Dx, device=device) * scale + 1.
        log_var = torch.randn(Dx, device=device) * scale

        # Set requires_grad to True for leaf tensors
        bias.requires_grad_(True)
        linear.requires_grad_(True)
        log_var.requires_grad_(True)

        params.append((bias, linear, log_var))

    return params


def generate_data(model_params, T=5, device='cpu'):
    mu0, Sigma0, A, Q, C, R = model_params
    Dx = mu0.shape[0]
    Dy = R.shape[0]

    x_true = torch.zeros((T, Dx), device=device)
    y_true = torch.zeros((T, Dy), device=device)

    mvn_x = dist.MultivariateNormal(mu0, Sigma0)
    mvn_y = dist.MultivariateNormal(torch.zeros(Dy, device=device), R)

    for t in range(T):
        if t > 0:
            mvn_x = dist.MultivariateNormal(torch.matmul(A, x_true[t - 1]), Q)
            x_true[t] = mvn_x.sample()
        else:
            x_true[0] = mvn_x.sample()

        mvn_y.loc = torch.matmul(C, x_true[t])
        y_true[t] = mvn_y.sample()

    return x_true, y_true


def log_marginal_likelihood(model_params, T, y_true):
    mu0, Sigma0, A, Q, C, R = [param.cpu().numpy() for param in model_params]
    y_true_np = y_true.cpu().numpy()
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
        yt = y_true_np[t, :] - np.dot(C, xpred)
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
        sign, logdet = torch.slogdet(Sigma)
        log_norm = -0.5 * dim * torch.log(torch.tensor(2. * torch.pi)) - 0.5 * logdet
        Prec = torch.inverse(Sigma)
        return log_norm - 0.5 * torch.sum((x - mu) @ Prec @ (x - mu).T, dim=1)

    def log_prop(self, t, Xc, Xp, y, prop_params, model_params):
        mu0, Sigma0, A, Q, C, R = model_params
        mut, lint, log_s2t = prop_params[t]
        s2t = torch.exp(log_s2t)

        if t > 0:
            mu = mut + torch.matmul(A, Xp.T).T * lint
        else:
            mu = mut + lint * mu0

        return self.log_normal(Xc, mu, torch.diag(s2t))

    def log_target(self, t, Xc, Xp, y, prop_params, model_params):
        mu0, Sigma0, A, Q, C, R = model_params
        if t > 0:
            logF = self.log_normal(Xc, torch.matmul(A, Xp.T).T, Q)
        else:
            logF = self.log_normal(Xc, mu0, Sigma0)
        logG = self.log_normal(y[t], torch.matmul(C, Xc.T).T, R)
        return logF + logG

    def log_weights(self, t, Xc, Xp, y, prop_params, model_params):
        return self.log_target(t, Xc, Xp, y, prop_params, model_params) - \
               self.log_prop(t, Xc, Xp, y, prop_params, model_params)

    def sim_prop(self, t, Xp, y, prop_params, model_params, device):
        mu0, Sigma0, A, Q, C, R = model_params
        mut, lint, log_s2t = prop_params[t]
        s2t = torch.exp(log_s2t)

        if t > 0:
            mu = mut + torch.matmul(A, Xp.T).T * lint
        else:
            mu = mut + lint * mu0
        return mu + torch.randn_like(Xp, device=device) * torch.sqrt(s2t)


if __name__ == '__main__':
    # Model hyper-parameters
    T = 5
    Dx = 5
    Dy = 3
    alpha = 0.42
    r = 0.1
    obs = 'sparse'

    # Training parameters
    param_scale = 0.5
    num_epochs = 10000
    step_size = 0.001
    N = 10

    # Set device for PyTorch (use 'cuda' for GPU, 'cpu' for CPU)
    device = 'cuda'

    # Initialize model and proposal parameters
    model_params = init_model_params(Dx, Dy, alpha, r, obs, device=device)
    prop_params = init_prop_params(T, Dx, param_scale, device=device)

    # Generate data
    print("Generating data...")
    x_true, y_true = generate_data(model_params, T, device=device)

    # Compute log marginal likelihood
    lml = log_marginal_likelihood(model_params, T, y_true)
    print("True log-marginal likelihood: " + str(lml))

    lgss_smc_obj = lgss_smc(T, Dx, Dy, N)


    # Define training objective
    def objective(prop_params):
        return -vsmc_lower_bound(prop_params, model_params, y_true, lgss_smc_obj, device=device, adapt_resamp=True)


    combined_params = [param for tuple_params in prop_params for param in tuple_params]
    # Optimizer
    optimizer = Adam(combined_params, lr=step_size)

    # Training loop
    print("     Epoch     |    ELBO  ")
    loss_values = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = objective(prop_params)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            current_loss = -loss.item()
            loss_values.append(current_loss)
            print("{:15}|{:20}".format(epoch, current_loss))

    # Output results
    opt_prop_params = prop_params  # After optimization
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, marker='o', linestyle='-', color='b')
    plt.title("Training ELBO Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Negative ELBO Loss")
    plt.grid(True)
    plt.show()
