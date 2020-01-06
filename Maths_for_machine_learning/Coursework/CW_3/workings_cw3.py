import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
np.set_printoptions(precision=5)

def create_design_matrix(K,type,basis,X_train):

    # Determine whether we are using test or eval data
    if type == 'train':
        n = len(X_train)
        X_phi = X_train
    else:
        n = N_test
        X_phi = X_test

    # Build design matrix
    assert basis in ['polynomial','trigonometric', 'exponential'], "Invalid basis (check spelling)"

    if basis == 'polynomial':
        Phi = np.zeros((n,K+1))

        for K in range(K+1):
            Phi[:,K] = X_phi.reshape(n,)**K

    elif basis == 'trigonometric':
        Phi = np.ones((n,2*K+1))

        for K in range(1,K+1):
            Phi[:,2*K-1] = np.sin(2*np.pi*K*X_phi.reshape(n,))
            Phi[:,2*K] = np.cos(2*np.pi*K*X_phi.reshape(n,))

    elif basis == 'exponential':
        Phi = np.ones((n,K+1))
        l = 0.1
        mus = np.linspace(-0.5,1,K)

        for K in range(1,K+1):
            Phi[:,K] = np.exp(-(np.squeeze(X_phi)-mus[K-1])**2/(2*l**2))

    return Phi

def get_theta_ml(K, basis, X_train, Y_train):

    # print(X_train, Y_train)
    Phi_train = create_design_matrix(K,'train',basis,X_train)       # Create design matrix for train set

    theta_ml = np.linalg.inv(Phi_train.T@Phi_train)@Phi_train.T@Y_train

    Phi_test = create_design_matrix(K,'test',basis,X_train)     # Create design matrix for test set
    Y_test = Phi_test@theta_ml
    return Y_test, theta_ml

def get_sigma_sq_ml(K,theta_ml,basis,X_train,Y_train):
    Phi = create_design_matrix(K,'train',basis,X_train)
    sigma_sq_ml = (1/len(X_train))*(Y_train - Phi@theta_ml).T@(Y_train - Phi@theta_ml)
    return np.squeeze(sigma_sq_ml)

def mle(basis, Ks, X_train, Y_train):
    '''
    Get MLE estimate and plots
    '''

    global X_test
    if basis == 'polynomial':
        X_test = np.reshape(np.linspace(-0.3,1.3,N_test), (N_test,1))
    elif basis == 'trigonometric':
        X_test = np.reshape(np.linspace(-1,1.2,N_test), (N_test,1))

    ml_estimators = []      # (theta_ml, sigma_sq_ml)
    for K in Ks:
        # Plot mean values
        Y_test, theta_ml = get_theta_ml(K, basis, X_train, Y_train)
        plt.plot(X_test,Y_test,label=f'K = {K}')

        sigma_sq_ml = get_sigma_sq_ml(K, theta_ml, basis, X_train, Y_train)      # Get variance

        ml_estimators.append((theta_ml, sigma_sq_ml))       # Save output

    plt.scatter(X,Y,10,'black')
    plt.legend()
    if basis == 'polynomial':
        plt.ylim(-2,10)
    plt.xlabel('x')
    plt.ylabel('y, predicted mean')
    # plt.show()
    # plt.savefig(f'plots/{basis}.pdf', bbox_inches='tight')
    plt.close()
    [print(i[1]) for i in ml_estimators]

def leave_one_out(basis, Ks, X_train, Y_train):
    '''
    Use leave-one-out cross validation to estimate parameters
    '''

    global X_test
    X_test = np.reshape(np.linspace(-1,1.2,N_test), (N_test,1))

    errors, sigma_sq_mls = [],[]      # (theta_ml, sigma_sq_ml)
    for K in Ks[:]:
        es = []
        for omit in range(25):
            X_t = np.vstack((X_train[:omit], X_train[omit+1:]))
            Y_t = np.vstack((Y_train[:omit], Y_train[omit+1:]))
            X_val = X_train[omit]
            Y_val = Y_train[omit]

            Y_test, theta_ml = get_theta_ml(K, basis, X_t, Y_t)

            # Compute squared avg test error
            Phi_val = create_design_matrix(K,'train',basis,X_val)
            Y_val_pred = Phi_val@theta_ml
            error = abs(Y_val - Y_val_pred)**2
            es.append(error)

        # Get variance
        Y_test, theta_ml = get_theta_ml(K, basis, X_train, Y_train)
        sigma_sq_ml = get_sigma_sq_ml(K, theta_ml, basis, X_train, Y_train)

        errors.append(np.mean(es))
        sigma_sq_mls.append(sigma_sq_ml)

    return errors, sigma_sq_mls

def q1_ab():
    '''
    Generate required plots for parts a and b
    '''
    mle('polynomial', [0,1,2,3,11], X, Y)
    mle('trigonometric', [1,11], X, Y)

def q1_c():
    basis = 'trigonometric'
    rn = range(11)
    errors, sigma_sq_mls = leave_one_out(basis, rn, X, Y)

    plt.plot(rn, errors, label='Squared avg test error')
    plt.plot(rn, sigma_sq_mls, label='MLE variances')
    plt.legend()
    plt.xlabel('Order of basis')
    plt.ylabel('Test error, MLE variance')
    # plt.show()
    plt.savefig(f'plots/sq_error_variance.pdf', bbox_inches='tight')
    plt.close()

def lml(alpha, beta, Phi, Y):
    '''
    Returns the log-marginal likelihood
    '''
    N = len(Y)
    Sigma = alpha*Phi@Phi.T + beta*np.eye(len(Phi))
    lml = -(N/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(Sigma)) - 0.5*Y.T@np.linalg.inv(Sigma)@Y
    return lml

def test_lml():
    '''
    Method to test lml and grad_lml
    '''
    Phi = create_design_matrix(1,'train','polynomial',X)
    # print(Phi)
    # J = grad_lml(1,2,Phi,Y)
    J = lml(1,2,Phi,Y)
    return J

def grad_lml(alpha,beta,Phi,Y):
    '''
    Returns gradient of the log-marginal likelihood wrt vector [alpha, beta]
    '''
    Sigma = alpha*Phi@Phi.T + beta*np.eye(len(Phi))
    Sigma_inv = np.linalg.inv(Sigma)
    J1 = -0.5*np.trace(Sigma_inv@Phi@Phi.T) + np.squeeze(0.5*Y.T@Sigma_inv@Phi@Phi.T@Sigma_inv@Y)
    J2 = -0.5*np.trace(Sigma_inv) + np.squeeze(0.5*Y.T@Sigma_inv@Sigma_inv@Y)
    return -np.array([J1, J2])

def grad_descent(X, Y, alpha_start=0.5, beta_start=0.8, iterations=1000, eta=0.005, features=1, basis='polynomial'):

    # Perform GD
    x = np.array([alpha_start, beta_start])
    xs = [x.copy()]
    Phi = create_design_matrix(features, 'train', basis, X)
    grads = [grad_lml(x[0], x[1], Phi, Y)]
    for i in range(iterations):
        x -= eta*grad_lml(x[0], x[1], Phi, Y)
        xs.append(x.copy())
        grads.append(grad_lml(x[0], x[1], Phi, Y))
    # [print(i) for i in xs]

    return grads, xs

def q_2b(X, Y, eta=None, alpha_start=0.0, beta_start=0.3, alpha_end=3, beta_end=1):

    # Make coutour plot
    alpha = np.linspace(alpha_start,alpha_end,100)
    beta = np.linspace(beta_start,beta_end,100)
    mx1, mx2 = np.meshgrid(alpha, beta)

    Phi = create_design_matrix(1, 'train', 'polynomial', X)
    fs = np.array([lml(x1,x2,Phi,Y) for x1,x2 in zip(mx1.ravel(), mx2.ravel())]).reshape(len(mx1),-1)
    # print(fs)
    grads, xs = grad_descent(X, Y, alpha_start=0.5, beta_start=0.8, iterations=1000, eta=0.005, features=1, basis='polynomial')

    fig = plt.contour(alpha, beta, fs, levels=30)
    # cbar = plt.colorbar(fig)
    # Add path of grad descent
    x1s = [x[0] for x in xs]
    x2s = [x[1] for x in xs]
    plt.scatter(x1s, x2s, 5, 'red')
    plt.plot(x1s, x2s, 'red')

    # Show
    plt.xlabel('alpha')
    plt.ylabel('beta')
    # plt.title(f'Contour plot of log marginal likelihood with gradient ascent')
    # plt.show()
    plt.savefig('plots/log_marginal_likelihood.pdf', bbox_inches='tight')
    plt.close()

    return x1s[-1], x2s[-1]

def q_2c(X, Y):
    basis = 'trigonometric'
    max_vals = []
    for K in range(13):
        grads, xs = grad_descent(X, Y, alpha_start=0.1, beta_start=0.1, iterations=10000, eta=0.00001, features=K, basis=basis)
        Phi = create_design_matrix(K, 'train', basis, X)
        max_vals.append(np.squeeze(lml(xs[-1][0],xs[-1][1],Phi,Y)))
        print(f'iteration {K}/12 done')

    [print(m) for m in max_vals]

    plt.plot(range(13),max_vals, 'red')
    plt.xlabel('Order of Basis')
    plt.ylabel('Max log marginal likelihood')
    # plt.show()
    plt.savefig('plots/basis_vs_max_mar_likeli.pdf', bbox_inches='tight')
    plt.close()

def q_2d(X, Y):
    K, basis = 10, 'exponential'
    Phi = create_design_matrix(K, 'train', basis, X)
    # print(Phi)

    # Posterior over weights:
    alpha, beta = 1, 0.1
    S_n = np.linalg.inv((1/alpha)*np.eye(len(Phi.T)) + (1/beta)*Phi.T@Phi)
    m_n = (1/beta)*S_n@Phi.T@Y
    samples = np.random.multivariate_normal(mean=np.squeeze(m_n), cov=S_n, size=5)

    global X_test
    X_test = np.reshape(np.linspace(-1,1.5,N_test), (N_test,1))
    Phi_test = create_design_matrix(K, 'test', basis, X)
    for n,s in enumerate(samples):
        y_star = s@Phi_test.T
        plt.plot(X_test, y_star,label=f'sample {n}',alpha=0.3)

    # Compute mean function (MAP estimate)
    y_star = np.squeeze(m_n.T@Phi_test.T)
    plt.plot(X_test, y_star,'red',label='mean function')

    # Compute error bars
    # err = Phi_test.T@np.linalg.inv((1/alpha)*np.eye(len(Phi_test.T)) + (1/beta)*Phi_test.T@Phi_test)@Phi_test
    err_bars = np.zeros((len(Phi_test),1))
    err_shaded = np.zeros((len(Phi_test),1))
    for n,i in enumerate(Phi_test[:]):
        i = i.reshape(1,-1)
        err_bars[n] = 2*(i@S_n@i.T + 0.1)**0.5
        err_shaded[n] = 2*(i@S_n@i.T)**0.5
        # print(i.shape, i.T.shape)
        # print(i@S_n@i.T)
    # Shaded area
    s1 = y_star + np.squeeze(err_shaded)
    s2 = y_star - np.squeeze(err_shaded)
    # plt.plot(X_test, s1,'--',color='blue')
    # plt.plot(X_test, s2,'--',color='blue')
    plt.fill_between(np.squeeze(X_test), s1, s2, facecolor='red', alpha=0.1, label='errors excl. noise')
    # Error bars
    err_bar_1 = y_star + np.squeeze(err_bars)
    err_bar_2 = y_star - np.squeeze(err_bars)
    plt.plot(X_test, err_bar_1,'--',color='black',linewidth=1, label = 'errors inc. noise')
    plt.plot(X_test, err_bar_2,'--',color='black',linewidth=1)
    # print(y_star.shape, errs.shape)
    # print(err_bar_1)

    # print(samples.shape)

    plt.scatter(X,Y,10,'black')
    plt.legend(fontsize=7)
    plt.ylim(-3.5,3.5)
    # plt.xlim(-1.5,2)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.show()
    plt.savefig(f'plots/exp_with_errs.pdf', bbox_inches='tight')
    plt.close()

    # Not sure about this
    # theta_ml = np.linalg.inv(Phi.T@Phi)@Phi.T@Y
    # sigma_sq_ml = get_sigma_sq_ml(None, theta_ml, 'exponential', X, Y)
    # conf_int = 2*sigma_sq_ml**0.5
    # print(conf_int)

def main():
    # q1_ab()
    q1_c()
    # print(test_lml())
    # grad_descent(X,Y)
    # q_2b(X, Y)
    # q_2c(X,Y)
    q_2d(X,Y)

if __name__ == '__main__':

    # np.random.seed(123)

    N = 25
    X = np.reshape(np.linspace(0,0.9,N,dtype=np.float_), (N,1))
    Y = np.cos(10*X**2) + 0.1*np.sin(100*X)
    N_test = 400
    # [print(x, y) for x,y in zip(X,Y)]
    # print(len(X))

    main()
