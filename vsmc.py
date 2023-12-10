import torch


def resampling(w, device):
    with torch.no_grad():  # Ensures no gradients are computed in this block
        N = w.shape[0]
        bins = torch.cumsum(w, dim=0)
        ind = torch.arange(N, dtype=torch.float32, device=device)
        u = (ind + torch.rand(N, device=device)) / N  # Using torch.rand for uniform sampling
        return torch.searchsorted(bins, u, right=True)


def vsmc_lower_bound(prop_params, model_params, y, smc_obj, device, verbose=False, adapt_resamp=False):
    T = y.shape[0]
    Dx = smc_obj.Dx
    N = smc_obj.N

    X = torch.zeros((N, Dx), device=device)
    Xp = torch.zeros((N, Dx), device=device)
    logW = torch.zeros(N, device=device)
    W = torch.exp(logW)
    W /= torch.sum(W)
    logZ = 0.0
    ESS = 1.0 / torch.sum(W ** 2) / N

    for t in range(T):
        # Resampling
        if adapt_resamp:
            if ESS < 0.5:
                ancestors = resampling(W, device=device)
                Xp = X[ancestors]
                logZ = logZ + max_logW + torch.log(torch.sum(W)) - torch.log(torch.tensor(N, dtype=torch.float32))
                logW = torch.zeros(N, device=device)
            else:
                Xp = X
        else:
            if t > 0:
                ancestors = resampling(W, device=device)
                Xp = X[ancestors]
            else:
                Xp = X

        # Propagation
        X = smc_obj.sim_prop(t, Xp, y, prop_params, model_params, device=device)

        # Weighting
        if adapt_resamp:
            logW = logW + smc_obj.log_weights(t, X, Xp, y, prop_params, model_params)
        else:
            logW = smc_obj.log_weights(t, X, Xp, y, prop_params, model_params)

        max_logW = torch.max(logW)
        W = torch.exp(logW - max_logW)
        if adapt_resamp:
            if t == T - 1:
                logZ = logZ + max_logW + torch.log(torch.sum(W)) - torch.log(torch.tensor(N, dtype=torch.float32))
        else:
            logZ = logZ + max_logW + torch.log(torch.sum(W)) - torch.log(torch.tensor(N, dtype=torch.float32))
        W = W / torch.sum(W)
        ESS = 1.0 / torch.sum(W ** 2) / N
    if verbose:
        print('ESS: ' + str(ESS))
    return logZ


def sim_q(prop_params, model_params, y, smc_obj, device, verbose=False):
    T = y.shape[0]
    Dx = smc_obj.Dx
    N = smc_obj.N

    X = torch.zeros((N, T, Dx))
    logW = torch.zeros(N)
    W = torch.zeros((N, T))
    ESS = torch.zeros(T)

    for t in range(T):
        # Resampling
        if t > 0:
            ancestors = resampling(W[:, t - 1], device=device)
            X[:, :t, :] = X[ancestors, :t, :]

        # Propagation
        X[:, t, :] = smc_obj.sim_prop(t, X[:, t - 1, :], y, prop_params, model_params, device=device)

        # Weighting
        logW = smc_obj.log_weights(t, X[:, t, :], X[:, t - 1, :], y, prop_params, model_params)
        max_logW = torch.max(logW)
        W[:, t] = torch.exp(logW - max_logW)
        W[:, t] /= torch.sum(W[:, t])
        ESS[t] = 1.0 / torch.sum(W[:, t] ** 2)

    # Sample from the empirical approximation
    bins = torch.cumsum(W[:, -1], dim=0)
    u = torch.rand()
    B = torch.searchsorted(bins, u)

    if verbose:
        print('Mean ESS', torch.mean(ESS) / N)
        print('Min ESS', torch.min(ESS))

    return X[B, :, :]
