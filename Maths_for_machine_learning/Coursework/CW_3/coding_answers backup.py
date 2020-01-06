import numpy as np
# from workings_cw3 import create_design_matrix

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

# if __name__=='__main__':
#     '''
#     For self testing
#     '''
#     N = 25
#     X = np.reshape(np.linspace(0,0.9,N,dtype=np.float_), (N,1))
#     Y = np.cos(10*X**2) + 0.1*np.sin(100*X)
#
#     Phi = create_design_matrix(3,'train','trigonometric',X)
#
#     a_bs = [(1,1),(5,2),(2,3)]
#     for a,b in a_bs:
#         alpha = a
#         beta = b
#         print(f'lml = {lml(alpha, beta, Phi, Y)}')
#         print(f'grad_lml = {grad_lml(alpha, beta, Phi, Y)}')
