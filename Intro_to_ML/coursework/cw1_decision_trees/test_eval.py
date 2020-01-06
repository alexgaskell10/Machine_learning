from classes.DecisionTree import DecisionTree
from classes.Evaluator import DecisionTreeEvaluator
import numpy as np


np.random.seed(100)
data_path = "wifi_db/clean_dataset.txt"
eval = DecisionTreeEvaluator(data_path, k_folds=10, max_depth=100)
eval.run_cross_validation()
