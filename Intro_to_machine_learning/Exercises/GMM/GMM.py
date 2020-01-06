import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as  plt
from sklearn.preprocessing import minmax_scale
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
plt.style.use('seaborn')

class GMM:

    def __init__(self, data, data_features, k=3):
        self.k = k
        self.data = data[:,:data_features]
        self.labels = data[:,data_features:]

        # Create Gaussians for fitting GMM
        self.gaussians = []
        self._init_gaussians()

        # Create responsibilities matrix
        self.rs = np.random.rand(len(self.data), self.k)

        # Init weights
        self.pis = np.full((self.k,), 1/self.k)

    def _init_gaussians(self):
        'Create K random Gaussians'
        for i in range(self.k):
            mean = self.data[np.random.randint(len(self.data))]
            cov = np.eye(len(mean))
            G = multivariate_normal(mean, cov)
            self.gaussians.append(G)

    def e_step(self):
        'Compute responsibilities'
        for i in range(self.k):
            self.rs[:,i] = self.pis[i] * self.gaussians[i].pdf(self.data)
        self.rs /= np.sum(self.rs, axis=1).reshape(-1,1)

    def m_step(self):
        'Update mean, covariances and weights of each Gaussian'
        for n,g in enumerate(self.gaussians):
            n_k = sum(self.rs[:,n].reshape(-1,1))
            # Mean = avg position of data weighted by responsibilites
            g.mean = np.sum(self.data * self.rs[:,n].reshape(-1,1), axis=0) / n_k
            # Cov = scatter of data weighted by responsibilites
            g.cov = np.sum(np.array([self.rs[i,n] * np.outer(x-g.mean,x-g.mean) for i,x in enumerate(self.data)]), axis=0) / n_k
            # Update weights of gaussians
            self.pis[n] = n_k / len(self.data)

    def plot_points(self):
        x1_dim = 2
        x2_dim = 3

        fig = plt.figure(num=1, figsize=(14, 6), facecolor='w', edgecolor='k')
        ax1, ax2 = fig.add_subplot(1,2,1), fig.add_subplot(122, projection='3d')

        # Plot data
        xs = self.data[:, x1_dim]
        ys = self.data[:, x2_dim]
        colors = minmax_scale(self.rs, feature_range=(0.1,0.9), axis=1)
        ax1.scatter(xs, ys, 5, c=colors)

        # Plot centroids and desities of gaussians
        cols = [(1,0.1,0.1), (0,0.8,0.3), (0.1,0.1,1)]
        z = np.zeros((300,300))
        # x1 = np.linspace(3, 8, 300)
        # x2 = np.linspace(1, 5, 300)
        x1 = np.linspace(-1, 7, 300)
        x2 = np.linspace(-1, 5, 300)
        X1, X2 = np.meshgrid(x1, x2)
        pos = np.empty(X1.shape + (2,))
        pos[:, :, 0] = X1; pos[:, :, 1] = X2

        for n,g in enumerate(self.gaussians):

            marginal = multivariate_normal(g.mean[[x1_dim, x2_dim]], g.cov[[x1_dim, x2_dim], [x1_dim, x2_dim]])
            fs = marginal.pdf(pos)
            z += fs * self.pis[n]

            ax1.scatter(g.mean[x1_dim], g.mean[x2_dim], c=[cols[n]], alpha=1, s=150*self.pis[n])
            ax1.contour(x1, x2, fs, levels=5, colors=[cols[n]], alpha=0.3)

        surf = ax2.plot_surface(X1, X2, z, rstride=8, cstride=8, alpha=0.8, cmap=cm.ocean)

        ax2.set_xlabel('$x1$')
        ax2.set_ylabel('$x2$')
        ax2.set_zlabel('$p(x1,x2)$')
        # ax1.axis([3,8,1,5])
        ax1.set_xlabel('$x1$')
        ax1.set_ylabel('$x2$')

if __name__ == '__main__':
    d = np.genfromtxt('datasets/iris.dat')      # Input path of data file here
    feature_cols = 4                            # input features of data here
    model = GMM(d, feature_cols)
    for i in range(10):
        model.plot_points()
        model.e_step()
        model.m_step()
        plt.draw()
        plt.pause(100**-1)
        if i < 9:
            plt.close()
        else:
            plt.show()
