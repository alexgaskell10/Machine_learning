import numpy as np
from .DecisionTree import DecisionTree
class DecisionTreeEvaluator:
    '''
    Class used to perform k-fold cross-validation on the tree for a given dataset
    Also contains helper methods which calculate metrics of interest
    '''

    def __init__(self, data_path, k_folds=10, max_depth=10, prune_tree=True):
        self.K = k_folds
        self.max_depth = max_depth
        self.tree = DecisionTree(data_path,  max_depth)
        self.data = self.tree.data
        self.prune_tree = prune_tree # To prune or not to prune

    def run_cross_validation(self):
        '''
        Method to run k fold cross validation.

        Algo:
        - Init a decision tree object
        - Shuffle data and get the row indices to split into k folds
        - Outer loop:
            - Split the data into test (1 fold) and training & validation (k-1 folds)
            - Inner loop:
                - Split training (k-2 folds) and validation (1 fold)
                - Train the tree using training dataset
                - Evaluate accuracy of every sample in the validation set. Return the prediction and label
            - Repeat inner loop k-1 times, with validation set as a different fold each time
        - Repeat outer loop k times, with test set as a different fold each time
        '''
        np.random.shuffle(self.data)

        conf_mats = []
        accuracies_outer = []

        for test_set_fold in range(self.K):      # Outer loop (run k times)
            test = self.get_subset(test_set_fold)

            accuracies_inner = []
            count_val = 0
            # Inner loop (run k-1) times
            for validation_set_fold in range(self.K):
                if validation_set_fold == test_set_fold:
                    continue # can't be the same
                count_val += 1
                validation = self.get_subset(validation_set_fold)
                train = self.get_train_set(test_set_fold, validation_set_fold)
                tree = DecisionTree(max_depth=self.max_depth)
                # Train decision tree using (k-2) folds as training data
                tree.create_tree(train)

                if self.prune_tree:
                    # Prune tree on validation data
                    tree.prune(validation)

                # Test pruned tree
                num_samples = len(test)
                labels = test['room']
                prediction = np.zeros(num_samples).astype('int8')
                for i, s in enumerate(test):
                    prediction[i]  = tree.classify_observation(s)

                conf_mat = self.confusion_matrix(prediction, labels)
                conf_mats.append(conf_mat)

                correct = num_samples - np.count_nonzero(labels - prediction)
                accuracies_inner.append(correct/num_samples)

                print(f'\nTest set fold: {test_set_fold+1} \\ {self.K} \nValidation set fold: {count_val} \\ {self.K-1}')
                print(f'{correct} / {num_samples} correct \n')

            accuracies_outer.append(accuracies_inner)

        avg_conf_mats = np.mean(conf_mats, axis=0)
        avg_prec = self.precision(avg_conf_mats)
        avg_rec = self.recall(avg_conf_mats)
        avg_F_1 = self.F_1(avg_conf_mats)

        print(avg_conf_mats)
        print(avg_prec)
        print(avg_rec)
        print(avg_F_1)

        print([np.mean(a) for a in accuracies_outer])
        print([np.mean(accuracies_outer)], np.std(accuracies_outer))

    def get_subset(self, fold_index):
        subset_size = int(len(self.data)/self.K) # could cause issues for odd numbers
        set_start = fold_index*subset_size
        return self.data[set_start : set_start + subset_size]

    def get_train_set(self, test_set_fold, validation_set_fold):
        # indices between 0 and K not taken by test or validation are training
        first_fold = True
        for i in range(self.K):
            if i in [test_set_fold, validation_set_fold]:
                continue
            if first_fold:
                train = self.get_subset(i)
                first_fold = False
            else:
                train = np.append(train, self.get_subset(i))
        return train


    def confusion_matrix(self, prediction, labels):
        '''
        Creates a confusion matrix given a dataset and its predictions.
        :input evals: list of tuples containing (prediction, label)
        :return conf_mat: the confusion matrix for the tree. The rows are
        the labels; columns are the predictions
        '''
        evals = list(zip(list(labels),list(prediction)))
        labels_unique = sorted(np.unique(labels))

        counter = []
        for i in labels_unique:
            for j in labels_unique:
                counter += [evals.count((i,j))]

        conf_mat = np.array(counter).reshape(len(labels_unique),len(labels_unique))
        conf_mat = conf_mat/conf_mat.sum(axis=1, keepdims=True)

        return conf_mat

    def recall(self, avg_conf_mats):
        '''
        Recall = TP / TP + FN (for each class)
        '''
        rec = []
        for i in range(len(avg_conf_mats)):
            rec += [avg_conf_mats[i,i] / sum(avg_conf_mats[i])]

        return np.array(rec)

    def precision(self, avg_conf_mats):
        '''
        Precision = TP / TP + FP (for each class)
        '''
        prec = []
        for i in range(len(avg_conf_mats)):
            prec += [avg_conf_mats[i,i] / sum(avg_conf_mats[:,i])]

        return np.array(prec)

    def F_1(self, avg_conf_mats):
        '''
        F1 = 2 * Recall * Precision / (Recall + Precision) (for each class)
        '''
        rec = self.recall(avg_conf_mats)
        prec = self.precision(avg_conf_mats)

        F_1 = []
        for i in range(len(rec)):
            F_1 += [2 * prec[i] * rec[i] / (prec[i] + rec[i])]

        return np.array(F_1)

    # def tree_analytics(self, data, evals):
    #     '''
    #     Method for getting metrics for a tree
    #     '''
    #     for e in evals[:]:
    #         conf_mat = self.confusion_matrix(data, evals[0])
    #         recall = self.recall(conf_mat)
    #         precision = self.precision(conf_mat)
    #         F_1 = self.F_1(conf_mat)
    #
    #     print(f'Confusion matrix = \n {conf_mat}')
    #     print(f'\n Recall = {recall}')
    #     print(f'\n Precision = {precision}')
    #     print(f'\n F_1 = {F_1} \n')
