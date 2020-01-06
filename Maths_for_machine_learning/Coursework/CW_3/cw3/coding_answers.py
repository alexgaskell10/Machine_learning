import numpy as np

def lml(alpha, beta, Phi, Y):
    '''
    Returns the log-marginal likelihood
    '''
    N = len(Y)
    Sigma = alpha*Phi@Phi.T + beta*np.eye(len(Phi))
    lml = -(N/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(Sigma)) - 0.5*Y.T@np.linalg.inv(Sigma)@Y
    return np.squeeze(lml)

def grad_lml(alpha, beta, Phi, Y):
    '''
    Returns gradient of the log-marginal likelihood wrt vector [alpha, beta]
    '''
    Sigma = alpha*Phi@Phi.T + beta*np.eye(len(Phi))
    Sigma_inv = np.linalg.inv(Sigma)
    J1 = -0.5*np.trace(Siglma_inv@Phi@Phi.T) + np.squeeze(0.5*Y.T@Sigma_inv@Phi@Phi.T@Sigma_inv@Y)
    J2 = -0.5*np.trace(Sigma_inv) + np.squeeze(0.5*Y.T@Sigma_inv@Sigma_inv@Y)
    return np.array([J1, J2])
