import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det

def f1(x):
    x = x.reshape(2,1)
    return x.T@B@x - x.T@x + a.T@x - b.T@x

def f2(x):
    x = x.reshape(2,1)
    return np.cos((x-b).T@(x-b)) + (x-a).T@B@(x-a)

def f3(x):
    x = x.reshape(2,1)
    t1 = -np.exp(-(x-a).T@(x-a))
    t2 = -np.exp(-(x-b).T@B@(x-b))
    t3 = 0.1*np.log(det(0.01*np.eye(len(x))+x@x.T))
    f = 1 + t1 + t2 + t3
    return f

def grad_f1(x):
    x = x.reshape(2,1)
    grad = 2*(x-c).T@C
    return grad.reshape(2,)

def grad_f2(x):
    x = x.reshape(2,1)
    grad = -2*np.sin((x-b).T@(x-b))*(x-b).T + 2*(x-a).T@B
    return grad.reshape(2,)

def grad_f3(x):
    x = x.reshape(2,1)
    t1 = -np.exp(-(x-a).T@(x-a))*(-2*(x-a).T)
    t2 = -np.exp(-(x-b).T@B@(x-b))*(-2*(x-b).T@B)
    t3 = (0.1*2/(x.T@x + 0.01))*x.T
    grad = t1 + t2 + t3
    return grad.reshape(2,)

def grad_descent(x1_start=0.3,x2_start=0.,iterations=50,eta=0.01,f=f2):

    # Pick function
    if f == f2:
        grad = grad_f2
        f = f2
    elif f == f3:
        grad = grad_f3
        f = f3
    else:
        grad = grad_f1
        f = f1

    # Perform GD
    x = np.array([x1_start,x2_start])
    xs = [x.copy()]
    grads = [grad(x)]
    for i in range(iterations):
        x -= eta*grad(x)
        xs.append(x.copy())
        grads.append(grad(x))

    # Compute f(x) at each value
    fs = [f(x) for x in xs]
    # [print(i) for i in fs]
    return grads, xs, fs

def make_contour(f, xs, eta, x1_start=-0.4, x1_end=0.75, x2_start=-0.25, x2_end=1.25):

    # Make coutour plot
    x1 = np.linspace(x1_start,x1_end,100)
    x2 = np.linspace(x2_start,x2_end,100)
    mx1, mx2 = np.meshgrid(x1, x2)
    fs = np.array([f(np.array([x1, x2])) for x1,x2 in zip(mx1.ravel(), mx2.ravel())]).reshape(mx1.shape[0],mx1.shape[1])
    plt.contour(x1, x2, fs, levels=20)

    # Add path of grad descent
    x1s = [x[0] for x in xs]
    x2s = [x[1] for x in xs]
    plt.scatter(x1s, x2s, 5, 'red')
    plt.plot(x1s, x2s, 'red')

    # Show
    plt.xlabel('x1')
    plt.ylabel('x2')
    if f == f2:
        fun = 'f2'
    elif f == f3:
        fun = 'f3'
    plt.title(f'Contour plot of {fun} (step size = {eta})')
    plt.show()

    # Save
    # if f==f2:
    #     fname = f'contour_f2.pdf'
    # elif f==f3:
    #     fname = f'contour_f3.pdf'
    # plt.savefig(fname, bbox_inches='tight')
    plt.close()

def main():

    # f = f2
    # eta = 0.18
    # grads, xs, fs = grad_descent(eta=eta,f=f)
    # make_contour(f, xs, x1_start=-0.5, x1_end=0.6, x2_start=-0.25, x2_end=1.25, eta=eta)

    f = f3
    eta = 10
    grads, xs, fs = grad_descent(eta=eta,f=f, iterations=50)
    make_contour(f, xs, eta=eta)

    # f = f3
    # eta = 0.03
    # grads, xs, fs = grad_descent(eta=eta,f=f)
    # make_contour(f, xs, x1_start=-1.1, x1_end=0.6, x2_start=-0.25, x2_end=2.5, eta=eta)
    # make_contour(f, xs, x1_start=-2.1, x1_end=1.5, x2_start=-0.25, x2_end=4, eta=eta)


if __name__ == '__main__':
    B = np.array([[4,-2],[-2,4]])
    a = np.array([[0],[1]])
    b = np.array([[-2],[1]])

    C = B - np.identity(2)
    c = -0.5*np.linalg.inv(C)@(a-b)
    c0 = -c.T@C@c

    main()
    # x = np.array([1,1])
    # print(grad_f1(x))
    # print(grad_f2(x))
    # print(grad_f3(x))

    # grads, xs, fs = grad_descent(eta=0.0075,f=f2)
    # grads, xs, fs = grad_descent(eta=0.02,f=f3)
    # plt.plot([i[0] for i in xs], label='x1')
    # plt.plot([i[1] for i in xs], label='x2')
    # plt.plot([i[0] for i in fs], label='f')
    # plt.legend()
    # plt.show()
