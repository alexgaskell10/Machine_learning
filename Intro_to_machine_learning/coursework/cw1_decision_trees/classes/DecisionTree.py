import numpy as np


class DecisionTree:
    ''' 
    DecisionTree class, constains code to 
        - create trees
        - prune trees
        - evaluate tree
        - print trees

    May optionally be instantiated directly with a dataset path
    '''

    def __init__(self, dataset_path=None, max_depth=100):
        if dataset_path:
            self.data = self.load_data(dataset_path)
        self.max_depth = max_depth

    def load_data(self, dataset_path):
        '''
        Load data function does what it says.

        Data is stored in a structured numpy array with 8 bit integers.
        Features given by "signal_1"...."signal_7", labels "room".
        '''
        data_lab = ["signal_" + str(x) for x in range(1, 8)] + ["room"]
        dtypes = list(zip(data_lab, [np.int8]*8))
        data = np.genfromtxt(dataset_path, dtype=dtypes)
        return data

     ## ------ TREE CREATION ------ ##

    def create_tree(self, data=None):
        ''' 
        Tree creation function

        Either takes a dataset directly, or uses a stored dataset
        '''
        if data is not None:
            self.tree, self.tree_depth = self.decision_tree_learning(data, 0)
        else:
            self.tree, self.tree_depth = self.decision_tree_learning(
                self.data, 0)

    def decision_tree_learning(self, data, depth):
        '''
        Recursive decision tree learning function

        Tree splits in ascending order i.e.:
        - if v <= split_value --> left
        - if v > split_value --> right

        Tree is a nested dictionary, with each node having properties:
        "attr" - signal label, "val" split value, "left"&"right" - children

        Leaf nodes have "attr" = 0, and "label_counts" = array of label counts
        (for pruning) and "val" = modal label in label counts

        :input: matrix containing dataset, max depth of tree
        :output: tree dictionary and maximum depth reached
        '''

        # Base case - reached max_depth, or splitting dataset contains only
        # one type of label
        if (depth == self.max_depth) or (len(set(data['room'])) == 1):
            label_counts = np.bincount(data['room'], minlength=5)
            modal_label = np.argmax(label_counts)
            return ({"attr": 0, "val": modal_label,
                     "left": None, "right": None,
                     "label_counts": label_counts},   depth)

        # Split via find split
        split_feature, split_index = self.find_split(data)

        # Sort data by feature
        data = np.sort(data, order=split_feature)

        # Split dataset
        l_dataset = data[:split_index]
        r_dataset = data[split_index:]

        # Create node with root as split value
        node = {"attr": split_feature,
                "val": data[split_feature][split_index-1],
                "left": 0, "right": 0}

        # Let's do the RE-CURSION again...
        l_branch, l_depth = self.decision_tree_learning(l_dataset, depth + 1)
        r_branch, r_depth = self.decision_tree_learning(r_dataset, depth + 1)

        node["left"] = l_branch
        node["right"] = r_branch

        return (node, max(l_depth, r_depth))

    def find_split(self, data):
        '''
        Applies find_split_point method to every column and returns the split
        which gives the largest entropy gain

        :returns (feature, index) corresponding to best split
        '''
        gains = {}
        for feature in data.dtype.names[:-1]:
            try:
                index, gain = self.find_split_feature(data, feature)
                gains[(feature, index)] = gain
            except ValueError:
                # reached small splitting set with overlapping features
                pass

        ideal_split = max(gains, key=gains.get)
        return ideal_split

    def find_split_feature(self, data, feature):
        '''
        Method to find all split points for a feature
        Algo
        - Sort by value (for a given attribute)
        - Find unique data labels for each value
        - Compute entropy gain for each of the splitting points
        - Return the splitting point which maximizes entropy gain
        '''

        # Sort feature array by feature column
        data_sort = np.sort(data, order=feature)

        # Find indices corresponding to points at which feature value changes
        diff = np.diff(data_sort[feature])
        split_points = {x+1: 0 for x in np.where(diff > 0)[0]}

        # Calculate the gains associated with splits around those points
        total_ent = self.entropy(data_sort)
        for split in split_points.keys():
            split_points[split] = self.gain(
                total_ent, data_sort[:split], data_sort[split:])

        best_split = max(split_points, key=split_points.get)

        # Return best split and associated gain
        return best_split, split_points[best_split]

    def gain(self, total_ent, l_data, r_data):
        # Calculate gain
        return total_ent - self.entropy_remainder(l_data, r_data)

    def entropy(self, data):
        # Calculate entropy - sum over all values in data set
        # Use change of base log_2(x) = ln(x) / ln(2)
        divide = len(data['room'])
        unique, counts = np.unique(data['room'], return_counts=True)
        return (-np.sum(counts * ((np.log(counts)-np.log(divide)) / np.log(2)))/divide)

    def entropy_remainder(self, l_data, r_data):
        # Calculate entropy remainder given left and right splitting sets
        left = len(l_data)
        right = len(r_data)
        tot = left + right
        left_term = (left / tot) * self.entropy(l_data)
        right_term = (right / tot) * self.entropy(r_data)
        return left_term + right_term

    ## ------ PRUNING ------ ##
    def prune(self, validation_data):
        '''
        Prune function

        Given a validation set, we will recursively prune our tree until
        we have milked it for the most additional CLASSIFICATION ACCURACY

        This applies in place to the current tree and does not copy / return
        '''
        self.validation_data = validation_data

        # Find a base-line classification accuracy before pruning starts
        self.best_validation = self.evaluate_tree()

        # Start recursive pruning
        self.prune_recursive(self.tree)
        self.tree_depth = 0
        self.update_tree_depth(self.tree, 0)
        return

    def prune_recursive(self, node):
        '''
        Recursive pruning function

        Go to leaf nodes, replace them, evaluate the tree -> if this improves
        the validation accuracy then we keep the change, otherwise, re-add leaf
        and recurse to next leaf node (bottom-up)
        '''
        if node['attr'] == 0:  # Reached leaf node
            return

        # Want to try pruning on children before on self
        self.prune_recursive(node['left'])
        self.prune_recursive(node['right'])

        # If both children are leaves
        if node["left"]["attr"] == 0 and node["right"]["attr"] == 0:
            # Save node info
            save_attr = node["attr"]
            save_val = node["val"]

            # Convert node into leaf
            label_counts = node["left"]["label_counts"] + \
                node["right"]["label_counts"]
            modal_class = np.argmax(label_counts)
            node['attr'] = 0
            node['val'] = modal_class

            # Evaluate pruned tree
            val_score = self.evaluate_tree()
            if val_score >= self.best_validation:  # improved the tree
                # Stay pruned and throw children to garbage collector! >:D
                node["left"] = None
                node["right"] = None
                node["label_counts"] = label_counts
                self.best_validation = val_score
            else:
                # Return to initial state
                node["attr"] = save_attr
                node["val"] = save_val
            return

    def update_tree_depth(self, node, depth):
        ''' 
        Recursive inner loop for get_tree_dimensions
        '''
        self.tree_depth = max(depth, self.tree_depth)
        if node["attr"] == 0:
            return
        self.update_tree_depth(node["left"], depth + 1)
        self.update_tree_depth(node["right"], depth + 1)
        return

    def count_nodes(self):
        return self.count_nodes_recurse(self.tree)

    def count_nodes_recurse(self, node):
        if node["attr"] == 0: # base case
            return 0 # don't count leaves
        count = 1
        count += self.count_nodes_recurse(node['left'])
        count += self.count_nodes_recurse(node['right'])
        return count



    ## ------ EVALUATION ------ ##
    def evaluate_tree(self, val_data=None):
        '''
        Tree evaluation function for use with pruning

        If no data is provided, we evaluate on existing data - for debugging
        :returns classification accuracy
        '''
        data = self.validation_data if val_data is None else val_data

        # Straight forward binary accuracy
        num_samples = len(data)
        labels = data['room']
        prediction = np.zeros(num_samples).astype('int8')
        for i, s in enumerate(data):
            prediction[i] = self.classify_observation(s)

        correct = num_samples - np.count_nonzero(labels - prediction)
        return correct / num_samples

    def classify_observation(self, observation):
        '''
        Classification function

        Calls recursive classification function to classify a single observation
        '''
        return self.classify_recurse(self.tree, observation)

    def classify_recurse(self, node, observation):
        if node['attr'] == 0:  # Reached leaf node
            return node['val']  # Modal class

        if observation[node['attr']] <= node['val']:  # go left if <=
            return self.classify_recurse(node['left'], observation)
        else:  # go right
            return self.classify_recurse(node['right'], observation)

     ## ------ PRINTING ------ ##
    def print_tree(self):
        # Call recursive print function
        self.print_node(self.tree, 0)

    def print_node(self, node, depth):
        if node["attr"] == 0:
            print(depth * "-" + "Node " + str(node["val"]))
            print(node['label_counts'])
            return
        print(depth * "-" + "%s > %i" % (node["attr"], node["val"]))
        self.print_node(node["left"], depth+1)
        self.print_node(node["right"], depth+1)
