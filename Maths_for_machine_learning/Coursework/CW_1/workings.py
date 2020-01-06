import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.stats as stats
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import subprocess
import shlex
import scipy.special

'''1. Compute sample mean and sample cov matrix for the following dataset
(use 1/N) for cov matrix)'''
def ex1():
    # Sample mean
    D1 = np.array([1,2,3]).reshape(3,1)
    D2 = np.array([-1,0,0]).reshape(3,1)
    D3 = np.array([-4,4,2]).reshape(3,1)
    D = [D1, D2, D3]
    mu = np.array((D1 + D2 + D3)/3)

    # Sample cov matrix
    # Cov_D = E[(D - mu)^2]
    #        = (1/N)sum(Di-mu)(Di - mu).T
    Cov_D = sum([(Di-mu)*(Di-mu).T for Di in D])/len(D)
    print(Cov_D, mu)

# Check for truth vs following line:
# print(np.cov(np.hstack((D1, D2, D3)), bias=True))

'''2. Generate 2 datasets {(x1, x2)n} of 100 data points each. The datasets
have mean mu = [-1, 1].T and marginal variances sigma_sq_1 = 2, sigma_sq_2 = 0.5.
Ensure the datasets you generate are different. Visualize the two datasets and
explain how you generated them so their shapes are different.'''
def ex2():
    # Begin by drawing 100 samples from standard normal Gaussian and perform
    # LTs on each to so they are from the corect distribution
    # samples = np.random.normal(size=(100,1))

    ## Need to derive the below formally
    # x1 = (2**0.5)*samples + (-1)
    # x2 = (0.5**0.5)*samples + 1

    # x1 = {'mu':-1, 'sigma':2**0.5, 'samples':(2**0.5)*samples + (-1)}
    # x2 = {'mu':1, 'sigma':0.5**0.5, 'samples':(0.5**0.5)*samples + (1)}
    #
    # for p in [x1, x2]:
    #     x_axis = np.linspace(p['mu'] - 3*p['sigma'],p['mu'] + 3*p['sigma'], 100)
    #     plt.plot(x_axis, stats.norm.pdf(x_axis, p['mu'], p['sigma']))
    #     plt.scatter(p['samples'].reshape(len(p['samples'],)), np.zeros(len(p['samples'])),
    #                 s=30, alpha=0.5, marker='+')
    #     plt.xlabel('x1, x2')
    #     plt.ylabel('p(x1), p(x2)')
    # plt.show()

    # print(np.mean(x1), np.var(x1))
    # print(np.mean(x2), np.var(x2))

    np.random.seed(123)
    # Define mean/covariance of a Gaussian
    p1 = {'i':'1', 'mean':[-1, 1], 'cov':[[2, 0], [0, 0.5]]}
    p2 = {'i':'2', 'mean':[-1, 1], 'cov':[[2, -0.5], [-0.5, 0.5]]}
    for p in [p1,p2]:
        # generate a mesh-grid for evaluating the pdf
        x, y = np.mgrid[-3*np.sqrt(p['cov'][0][0])+p['mean'][0]:3*np.sqrt(p['cov'][0][0])+p['mean'][0]:.1,\
                        -3*np.sqrt(p['cov'][1][1])+p['mean'][1]:3*np.sqrt(p['cov'][1][1])+p['mean'][1]:.1]
        # stack x-y coordinates
        pos = np.dstack((x, y))
        # generate Gaussian object
        gaussian = multivariate_normal(p['mean'], p['cov'])
        # evaluate the Gaussian pdf at the x-y coordinates
        z = gaussian.pdf(pos)
        print(z.shape)

        # plotting: 3D and 2D contour projection
        f = plt.figure(num=None, figsize=(14, 6), facecolor='w', edgecolor='k')
        ax = f.add_subplot(1,2,1, projection='3d')
        ax.plot_wireframe(x,y,z, color="grey", rstride=4, cstride=4, alpha=0.8)
        cset = ax.contour(x,y,z, zdir='z', offset=-0.04,  alpha=0.6, linewidths=2)
        ax.set_zlim3d(-0.04, 0.175) #gaussian.pdf(p['mean']))
        ax.set_xlabel('$x1$')
        ax.set_ylabel('$x2$')
        ax.set_zlabel('$p(x1,x2)$')
        # # # print to pdf
        # # fname ='gaussian3d.pdf'
        # # plt.savefig(fname, bbox_inches='tight')
        # # proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))


        # N samples from the Gaussian
        N = 100
        x1, x2 = np.random.multivariate_normal(p['mean'], p['cov'], N).T

        # plot the the contour on top of the samples
        xx = [p['mean'][0]+2*np.sqrt(p['cov'][0][0]), 0]

        levels = np.append(0, np.linspace(gaussian.pdf(xx), gaussian.pdf(p['mean']), 10))

        plt.subplot(1,2,2)
        plt.scatter(x1,x2, s=6)
        plt.contour(x, y, z, levels, linewidths=2)
        plt.xlim(-5,3)
        plt.ylim(-1,3)
        plt.xlabel('$x1$')
        plt.ylabel('$x2$')
        plt.tight_layout()

        fname = 'gaussian3d {}.pdf'.format(p['i'])
        # plt.savefig(fname, bbox_inches='tight')
        # # proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))
        plt.close()

    # plt.show()

'''3. a) Compute the posterior distribution on μ (derive your result) and plot it.
      b) What has changed from the prior to the posterior? Describe properties of
        the prior and the posterior.'''
def ex3():
    # Prior
    xs = np.linspace(0,1,100)
    # ys_prior = Beta(2,2)
    ys_prior = 6*xs*(1-xs)
    plt.plot(xs, ys_prior)

    # fname = 'beta_prior.pdf'
    # plt.savefig(fname, bbox_inches='tight')
    # # proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))
    # plt.close()

    # Posterior
    # ys_post = scipy.special.binom(20,6)*6*(xs**7)*((1-xs)**14)*6 # <-- only proprtional as havent calculated p(evidence) (to do)
    ys_post = stats.beta.pdf(xs, 8, 16)
    # ys_post prop to Beta(2+6,2+20-6) = Beta(8,16)
    plt.plot(xs, ys_post)
    plt.xlabel('$\u03BC$')
    plt.ylabel('$p(\u03BC)$')
    plt.legend(['prior: Beta(2,2)', 'posterior: Beta(8,16)'])

    fname = 'beta_distribs.pdf'
    plt.savefig(fname, bbox_inches='tight')
    # # proc=subprocess.Popen(shlex.split('lpr {f}'.format(f=fname)))
    plt.close()


    # Compare properties:
    # E(mu) = alpha/(alpha+beta)
    E_prior = 2 / (2+2)
    E_post = 8 / (8+16)
    # V(mu) = (alpha + beta)/(((alpha+beta)**2)*(alpha+beta+1))
    Var_prior = (2+2)/(((2+2)**2)*(2+2+1))
    Var_post = (8+16)/(((8+16)**2)*(8+16+1))
    # Mode where the pdf is maximised
    # Mode = (alpha - 1)/((alpha+beta-2))
    Mode_prior = (2-1)/((2+2-2))
    Mode_post = (8-1)/((8+16-2))
    print(Mode_prior, Mode_post)

if __name__ == '__main__':
    ex2()
