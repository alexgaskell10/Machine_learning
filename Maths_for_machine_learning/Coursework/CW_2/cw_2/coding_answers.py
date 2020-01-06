import numpy as np

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


if __name__ == '__main__':

    B = np.array([[4,-2],[-2,4]])
    a = np.array([[0],[1]])
    b = np.array([[-2],[1]])

    C = B - np.identity(2)
    c = -0.5*np.linalg.inv(C)@(a-b)
    c0 = -c.T@C@c

    # x = np.array([1,1])
    # print(grad_f1(x))
    # print(grad_f2(x))
    # print(grad_f3(x))
