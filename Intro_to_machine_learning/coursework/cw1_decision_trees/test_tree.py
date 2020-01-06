import numpy as np
import matplotlib.pyplot as plt
from classes.DecisionTree import DecisionTree
from classes.Visualizer import Visualizer

## -- CONFIGURATION -- ##
TREE_DEPTH = 100
DATA_PATHS = ["wifi_db/clean_dataset.txt","wifi_db/noisy_dataset.txt"]


## -- RUN SETUP -- ##
def main():
    # np.random.seed(123)

    clean_unpruned, clean_pruned, noisy_unpruned, noisy_pruned = [],[],[],[]
    tree_depths = [1,2,3,4,5,10,20,50]
    n = 2

    for TREE_DEPTH in tree_depths:
        for i in range(n):
            DATA_PATH = DATA_PATHS[0]
            tree = DecisionTree(DATA_PATH, TREE_DEPTH)
            np.random.shuffle(tree.data)

            evaluation = tree.data[:200]
            test = tree.data[200:400]
            train = tree.data[400:]

            tree.data = train
            tree.create_tree()
            accuracy = tree.evaluate_tree(test)
            clean_unpruned.append((tree.tree_depth, tree.count_nodes(), accuracy))
            # V = Visualizer(tree)
            # V.draw_tree()
            tree.prune(evaluation)
            accuracy = tree.evaluate_tree(test)
            clean_pruned.append((tree.tree_depth, tree.count_nodes(), accuracy))
            # V = Visualizer(tree)
            # V.draw_tree()

            DATA_PATH = DATA_PATHS[1]
            tree = DecisionTree(DATA_PATH, TREE_DEPTH)
            np.random.shuffle(tree.data)

            evaluation = tree.data[:200]
            test = tree.data[200:400]
            train = tree.data[400:]

            tree.data = train
            tree.create_tree()
            accuracy = tree.evaluate_tree(test)
            noisy_unpruned.append((tree.tree_depth, tree.count_nodes(), accuracy))
            # V = Visualizer(tree)
            # V.draw_tree()
            tree.prune(evaluation)
            accuracy = tree.evaluate_tree(test)
            noisy_pruned.append((tree.tree_depth, tree.count_nodes(), accuracy))
            # V = Visualizer(tree)
            # V.draw_tree()

    # print(clean_un, clean_pr, noisy_un, noisy_pr)

    # Plots
    names = ['clean unpruned', 'clean pruned', 'noisy unpruned', 'noisy pruned']
    markers = ['o','s','^','p']
    edgecolors = ["blue", "orange", "green", "red"]
    # Plot depth vs accuracy
    # horizontal_offsets = [-0.1, -0.05, 0.05, 1]
    # xlabels = tree_depths
    plt.rcParams.update({'errorbar.capsize': 2})
    plt.rcParams.update({'lines.markeredgewidth': 1})
    fig, ax  = plt.subplots()
    for n,ds in enumerate([clean_unpruned, clean_pruned, noisy_unpruned, noisy_pruned]):
        xs = range(len(tree_depths)*n)#[i + j for i, j in zip(tree_depths, [horizontal_offsets[n]]*len(tree_depths))]
        ys = [i[-1]*100 for i in ds]
        print(xs, ys)
        (_, caps, _) = ax.errorbar(xs, ys,label=names[n],marker=markers[n], mfc='none', mec=edgecolors[n], ls='none', capsize=0.3)
    #     for cap in caps:
    #         cap.set_markeredgewidth(1)
    plt.legend()
    plt.ylabel('Classification accuracy (%)')
    plt.xlabel('Imposed depth cap')
    ax.set_xticks(range(len(tree_depths)))
    ax.set_xticklabels(tree_depths)
    # plt.show()
    plt.savefig('max_depth_plot.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

    # # Plot num nodes vs accuracy
    # for n,ds in enumerate([clean_unpruned, clean_pruned, noisy_unpruned, noisy_pruned]):
    #     xs = [i[1] for i in ds]
    #     ys = [i[-1]*100 for i in ds]
    #     plt.scatter(xs, ys,label=names[n],marker=markers[n])
    # plt.legend()
    # plt.ylabel('Classification accuracy (%)')
    # plt.xlabel('Nodes in tree')
    # # plt.show()
    # plt.savefig('nodes_plot.pdf', bbox_inches='tight', pad_inches=0)
    # plt.close()

def eval(tree, set):
    num_samples = len(set)
    labels = set['room']
    prediction = np.zeros(num_samples).astype('int8')
    for i, s in enumerate(set):
        prediction[i]  = tree.classify_observation(s)

    correct = num_samples - np.count_nonzero(labels - prediction)
    return 100 * (correct / num_samples)


## -- BOILER PLATE -- ##
if __name__ == "__main__":
    main()
