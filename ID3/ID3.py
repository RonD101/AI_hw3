import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, label_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.label_names = label_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """

        counts = class_counts(rows, labels)
        impurity = 0.0

        # ====== YOUR CODE: ======
        amountOfSample = len(rows)
        labels_types = set()
        for label in labels:
            labels_types.add(label)
        for label_type in labels_types:
            p = counts[label_type] / amountOfSample
            impurity -= p * math.log2(p)
        # ========================

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        # ====== YOUR CODE: ======
        amountLeft = float(len(left_labels))
        amountRight = float(len(right_labels))
        totalAmount = amountLeft + amountRight
        info_gain_value = current_uncertainty - (amountLeft / totalAmount) * self.entropy(left, left_labels) - (amountRight / totalAmount) * self.entropy(right, right_labels)
        # ========================

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """

        # gain, true_rows, true_labels, false_rows, false_labels = None, None, None, None, None
        # =========== maybe should be np.ndarray ===========#
        gain, true_rows, true_labels, false_rows, false_labels = None, [], [], [], []
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # ====== YOUR CODE: ======
        for i, sample in enumerate(rows):
            if question.match(sample):
                true_rows.append(sample)
                true_labels.append(labels[i])
            else:
                false_rows.append(sample)
                false_labels.append(labels[i])
        gain = self.info_gain(true_rows, true_labels, false_rows, false_labels, current_uncertainty)
        # ========================

        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        # ====== YOUR CODE: ======
        questions = set()
        for sample in rows:
            for i, feature in enumerate(sample):
                questions.add(Question(i, i, feature))
        for question in questions:
            gain, true_rows, true_labels, false_rows, false_labels = \
                self.partition(rows, labels, question, current_uncertainty)
            if gain > best_gain or (gain == best_gain and question.column_idx > best_question.column_idx):
                best_gain = gain
                best_question = question
                best_false_rows, best_false_labels = false_rows, false_labels
                best_true_rows, best_true_labels = true_rows, true_labels
        # ========================

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        best_question = None
        true_branch, false_branch = None, None

        # ====== YOUR CODE: ======
        leaf = Leaf(rows, labels)
        if len(leaf.predictions) == 1 or len(rows) < self.min_for_pruning:
            return leaf
        best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = \
            self.find_best_split(rows, labels)
        if len(best_true_rows) == 0 or len(best_false_rows) == 0:
            return leaf  # For cases that we don't have a question that will increase our IG
        true_branch = self.build_tree(best_true_rows, best_true_labels)
        false_branch = self.build_tree(best_false_rows, best_false_labels)
        # ========================

        return DecisionNode(best_question, true_branch, false_branch)

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """

        # ====== YOUR CODE: ======
        self.tree_root = self.build_tree(x_train, y_train)
        # ========================

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """

        if node is None:
            node = self.tree_root
        prediction = None

        # ====== YOUR CODE: ======
        while not isinstance(node, Leaf):
            if node.question.match(row):
                node = node.true_branch
            else:
                node = node.false_branch
        maxLabelAmount = - math.inf
        for label in node.predictions:
            if node.predictions[label] > maxLabelAmount:
                maxLabelAmount = node.predictions[label]
                prediction = label
        # ========================

        return prediction

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """

        y_pred = None

        # ====== YOUR CODE: ======
        y_pred = []
        for sample in rows:
            y_pred.append(self.predict_sample(sample))
        # ========================

        return y_pred
