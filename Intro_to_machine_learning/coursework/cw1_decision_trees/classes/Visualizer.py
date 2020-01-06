import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from .DecisionTree import DecisionTree


class Visualizer:
    ''' 
    Class takes a tree and produces visualizations of the tree

    Required quite a bit of hand tuning because matplotlib is horrible,
    as such, no customizability besides output path
    '''

    def __init__(self, tree):
        self.tree_class = tree
        self.tree = tree.tree
        

    def draw_tree(self, save_path=None):
        ''' 
        Method produces plots and optionally saves these

        NOTE appearance optimized for saving to pdf
        '''
        plt.style.use('seaborn-dark')
        self.tree_depth = self.tree_class.tree_depth

        # Font definition for yes/no labels
        self.label_font = font = {'family': 'Lato',
                                  'color':  'grey',
                                  'weight': 'bold',
                                  'size': 10,
                                  }
        self.label_color = cm.get_cmap('Pastel2', 4)(range(4))
        self.feature_color = cm.get_cmap('Set2', 7)(range(7))
        self.box_height = 10
        self.box_width = 44

        # Figure out spacing of nodes for plot
        self.initialize_position_map()

        fig, ax = plt.subplots(1, figsize=(20, 15))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_ylim(-self.box_height*(self.tree_depth+1.5), -self.box_height/2)
        ax.set_xlim(-self.box_width, self.max_width + 2*self.box_width)

        # Recursively place nodes
        self.place_node(ax, self.tree, 0, (0, 0))

        plt.draw()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()

    def place_node(self, axes, node, depth, parent_pos, direction=None):
        '''
        Recursive function which places nodes in the plot

        Central purpose is to calclulate positions in x, y coords
        '''

        # X position based on depth, and number of nodes already placed
        x = self.get_xpos(depth)
        y = parent_pos[1] - self.box_height

        # Draw node at calculated position
        self.draw_node(axes, node, x, y, parent_pos, direction)

        # Recursion stops at child node
        if node["attr"] == 0:
            return self.tree_depth

        # Recurse to children
        self.place_node(axes, node["left"], depth+1, (x, y), direction='l')
        self.place_node(axes, node["right"], depth+1, (x, y), direction='r')

    def draw_node(self, axes, node, x, y, parent_pos, direction):
        # Handle arrow first
        if (parent_pos[1] != 0):  # No arrow connects to root
            # Calculate arrow start and end
            px, py = parent_pos
            dx = x - px  
            px -= self.box_width/4
            py = py - 0.4  # - .6
            dy = -self.box_height + 1.2

            # Put into lists for clarity
            arr_start = [px+self.box_width/2, py]
            arr_end = [arr_start[0]+dx, arr_start[1]+dy]

            # Draw arrow
            axes.arrow(*arr_start, dx, dy,
                       head_width=.4, color="grey")

            # Annotate arrow with Y/N
            if direction == 'r':  # right child -> yes
                t = axes.text(arr_end[0], arr_end[1]+1,
                              "Y", fontdict=self.label_font)
            else:                 # left child -> no
                t = axes.text(arr_end[0], arr_end[1]+1,
                              "N", fontdict=self.label_font)
            # Add a circular box to label
            t.set_bbox(dict(boxstyle="circle", facecolor='white',
                            alpha=0.3, edgecolor='white'))

        # Now draw node
        if node["attr"] == 0:  # Leaf node
            color = self.label_color[node["val"] - 1]
            props = dict(boxstyle='round', facecolor=color, alpha=.8)
            axes.text(x+2.5, y, r"Room " +
                      str(node["val"]), size=7, wrap=True, bbox=props)
        else:                  # Branch node
            color = self.feature_color[int(node["attr"][-1]) - 1]
            props = dict(boxstyle='square', facecolor=color, alpha=0.5)
            axes.text(x+2.5, y, r"Signal $" +
                      str(node["attr"][-1]) + r">" + str(node["val"]) + "$", size=6,
                      wrap=True, bbox=props)

    def get_xpos(self, depth):
        '''
        Function returns x position for a node on the figure, based on depth
        and how many other nodes have been placed on the row
        '''
        x = self.pos_map[depth][0]
        self.pos_map[depth][0] += self.pos_map[depth][1]
        return x

    def initialize_position_map(self):
        '''
        Function creates a "position map" which symmetrically spaces nodes 
        based on the node size and maximum width of the tree at some level
        '''
        self.get_tree_dimensions()
        self.pos_map = {}

        # Calculate maximum width in figure-space
        max_width = (max(self.tree_info.values()) + 2) * (self.box_width*1.1)
        self.max_width = max_width

        # Fill the position map with spacings calculated dependant on the
        # number of nodes in each layer
        for depth in range(self.tree_depth + 1):
            number_of_nodes = self.tree_info[depth]
            left_space = (max_width - number_of_nodes * self.box_width) / (
                number_of_nodes + 1)
            jump_space = left_space + self.box_width*1.05
            self.pos_map[depth] = [left_space, jump_space]

    def get_tree_dimensions(self):
        '''
        Function creates a dictionary which will store the number of nodes per
        layer of the tree. Used to create position map.
        '''
        tree_info = {x: 0 for x in range(self.tree_depth + 1)}
        self.explore_tree(tree_info, self.tree, 0)
        self.tree_info = tree_info

    def explore_tree(self, tree_info, node, depth):
        ''' 
        Recursive inner loop for get_tree_dimensions
        '''
        tree_info[depth] += 1
        if node["attr"] == 0:
            return

        self.explore_tree(tree_info, node["left"], depth + 1)
        self.explore_tree(tree_info, node["right"], depth + 1)
        return
