from main import Game
from time import time
import csv
import matplotlib.pyplot as plt

def record_times():

    times = []
    combos = [(2,2,2),(2,3,2),(3,3,2),(4,2,2),(5,2,2),(6,2,2),(7,2,2),(4,3,2),(4,4,2),(5,3,2),(8,2,2)]
    for m,n,k in combos:
        for ab_flag in [True,False]:
            game = Game(m,n,k)
            last_move = (1,1)
            game.player_positions.add(last_move)
            game.ab_flag = ab_flag
            start_time = time()
            game.get_computer_move_with_minimax(last_move)
            result = (m,n,k), ab_flag, time() - start_time
            print(result)
            times.append((result))

    print("")
    [print(t) for t in times]
    save_results(times)

def save_results(results):
    with open("results.txt", 'w') as f:
        f.write('mnk;ab_flag;time (s)\n')
        for x in results:
            f.write(f'{x[0]};{x[1]};{x[2]}\n')

def load_results():
    data = []
    with open("results.txt") as f:
        for row in f:
            r = row.split(';')
            r[0] = eval(r[0])
            r[1] = True if r[1] == 'True' else False
            r[2] = float(r[2])
            data.append(r)

    # [print(d) for d in data]
    return data

def plot(data):
    ys_true = [d[2] for d in data if d[1] == True]
    xs_true = [d[0][0] * d[0][1] for d in data if d[1] == True]
    ys_false = [d[2] for d in data if d[1] == False]
    xs_false = [d[0][0] * d[0][1] for d in data if d[1] == False]

    plt.scatter(xs_true, ys_true, label='With alpha-beta pruning')
    plt.scatter(xs_false, ys_false, label='Without')
    plt.yscale('log')
    plt.ylabel('Time for first move (seconds, log scale)')
    plt.xlabel('Size of board (m * n)')
    plt.title('Time to compute first (reply) move vs size of board \n (with and without alpha-beta pruning)')
    plt.legend()
    # plt.show()
    plt.savefig('plots/time_vs_board.pdf')

if __name__=='__main__':
    results = load_results()
    plot(results)
