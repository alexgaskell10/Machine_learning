import numpy as np

def compute_TD(V_s_t, V_s_t1, r_t_1, alpha, gamma):
    return alpha*(r_t_1 + gamma*V_s_t1 - V_s_t)

def check_s_0(alpha, gamma):
    # print(alpha, gamma)
    return alpha*(2+alpha*(gamma*(2+alpha*(gamma*(2-alpha)-1))-1))-alpha

def main_TD():
    # Create trace
    trace = [       # Format (V_s_t, V_s_t1, r_t_1)
    ('V_s_0', 'V_s_1', 1),
    ('V_s_1', 'V_s_0', 0),
    ('V_s_0', 'V_s_2', 1),
    ('V_s_2', 'V_s_2', 1),
    ('V_s_2', 'V_s_0', 1),
    ('V_s_0', 'V_s_2', 1),
    ]

    # Init values to zero
    vs = {'V_s_0':0, 'V_s_1':0, 'V_s_2':0}
    print(vs)

    gamma = 1
    alpha = 1/2

    for t,i in enumerate(trace[:]):
        vs[i[0]] += compute_TD(vs[i[0]], vs[i[1]], i[2], alpha, gamma)
        print(f'{i[0]}: {vs[i[0]]}')

    print(vs)
    print(f'hand-calc s_0: {alpha + check_s_0(alpha, gamma)}')

    # check = alpha*(2+alpha*(gamma*(2-alpha)-1))            # alpha + alpha*(1+gamma*alpha*(2-alpha) - alpha)
    # print(check)

def main_bellman():
    P = np.array([[0,1/3,2/3],[1,0,0],[1/3,0,1/3]])
    R = np.array([1,0,1])
    gamma = 1

    v = np.linalg.inv(np.eye(3)-gamma*P)@R
    print(v)

if __name__ == '__main__':
    main_TD()
    # main_bellman()
