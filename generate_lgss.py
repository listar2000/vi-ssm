import numpy as np


def generate_lgssm_data(A, C, Q, R, mu_0, Sigma_0, T):
    """
    Generate a dataset for a Linear Gaussian State Space Model.
    :param A: State transition matrix.
    :param C: Observation matrix.
    :param Q: Covariance matrix for state transition noise.
    :param R: Covariance matrix for observation noise.
    :param mu_0: Mean of initial state distribution.
    :param Sigma_0: Covariance of initial state distribution.
    :param T: Number of time steps.
    :return: Generated state and observation sequences.
    """
    state_dim = A.shape[0]
    obs_dim = C.shape[0]

    # Initialize state and observation arrays
    states = np.zeros((T, state_dim))
    observations = np.zeros((T, obs_dim))

    # Initial state
    states[0] = np.random.multivariate_normal(mu_0, Sigma_0)
    observations[0] = np.random.multivariate_normal(C @ states[0], R)

    # Generate data
    for t in range(1, T):
        states[t] = np.random.multivariate_normal(A @ states[t-1], Q)
        observations[t] = np.random.multivariate_normal(C @ states[t], R)

    return states, observations


if __name__ == '__main__':
    # Parameters for the LGSSM
    state_dim = 2
    obs_dim = 2
    T = 2  # Number of time steps
    seed = 2023

    # Randomly chosen matrices for demonstration
    # A = np.random.randn(state_dim, state_dim)  # State transition matrix
    # C = np.random.randn(obs_dim, state_dim)    # Observation matrix
    # Q = np.eye(state_dim) * 0.1               # State noise covariance
    # R = np.eye(obs_dim) * 0.1                 # Observation noise covariance
    # mu_0 = np.zeros(state_dim)                # Initial state mean
    # Sigma_0 = np.eye(state_dim)               # Initial state covariance

    # simple case (anity check)
    state_dim, obs_dim = 1, 1
    A = 0.5 * np.eye(1)
    C = np.eye(1)
    Q = np.eye(1)
    R = np.eye(1)
    mu_0 = np.zeros(1)
    Sigma_0 = np.eye(1)

    np.random.seed(seed)
    # Generate the data
    states, observations = generate_lgssm_data(A, C, Q, R, mu_0, Sigma_0, T)

    # Save the matrices and data to a file
    output_data = {
        "A": A,
        "C": C,
        "Q": Q,
        "R": R,
        "mu_0": mu_0,
        "Sigma_0": Sigma_0,
        "states": states,
        "observations": observations
    }

    np.savez('./data/lgssm_data_simple.npz', **output_data)
    print("data generated successfully...")
