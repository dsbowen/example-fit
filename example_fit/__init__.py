"""ExampleFit

ExampleFit is an explanation method which explains observations' predicted 
outcome by their proximity to example data points.

It transforms a feature space X into a weighting space W, where W_ij is the 
weight observation i attaches to example j. Weights are normalized such that 
all 0 < W_ij < 1, and W_i sums to 1 for all i. The more similar observation 
i and example j, the closer W_ij is to 1. 

The standard use is to fit a linear model (with no intercept) on the weight 
space.

We interpret ExampleFit as follows: The more similar observation i and 
example j, the closer the prediction f(x_i) is to Beta_j.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.model_selection import train_test_split


class ExampleFit():
    """
    Parameters
    ----------
    model : callable
        The model must have fit, predict, and score methods (see sklearn).
    alpha : scalar
        Smoothing parameter for computing weights.
    """
    def __init__(self, model, alpha=1e-6):
        self.model = model
        self.alpha = alpha
        self.cdist_args, self.cdist_kwargs = [], {}
        self.train_test_split_args, self.train_test_split_kwargs = [], {}

    def set_cdist_args(self, *args, **kwargs):
        """Set the cdist args and kwargs

        The distance between observations and examples is computed using the 
        scipy cdist function. These are the arguments passed into the 
        function. For example, this is where you would change the distance 
        metric.
        """
        self.cdist_args = args
        self.cdist_kwargs = kwargs
        return self

    def set_train_test_split_args(self, *args, **kwargs):
        """Set the train_test_split args and kwargs

        ExampleFit's fit_validate method selects examples to maximize test 
        (or validation) set score. It uses the sklearn train_test_split to 
        split the dataset into train and test/validation data. These are the 
        arguments passed into the function.
        """
        self.train_test_split_args = args
        self.train_test_split_kwargs = kwargs
        return self

    def fit(self, X, y, examples_X, examples_y=None, *args, **kwargs):
        """Train on full training dataset
        
        Parameters
        ----------
        X : numpy.array
            (n observations x p features) matrix.
        y : numpy.array
            (n observations) target array.
        examples_X : numpy.array
            (m examples x p features) matrix.
        examples_y : numpy.array (optional)
            (m examples) target array for the examples.

        Returns
        -------
        Fitted model.
        """
        self.examples_X, self.examples_y = examples_X, examples_y
        return self.model.fit(self.compute_weight(X), y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        """Predict

        Parameters
        ----------
        X : numpy.array
            (n observations x p features) matrix.

        Returns
        -------
        (n observations) numpy.array of predicted outputs.
        """
        if hasattr(self, 'validating') and self.validating:
            return self.model.predict(X, *args, **kwargs)
        return self.model.predict(self.compute_weight(X), *args, **kwargs)

    def example_predict(self, X):
        """Predict based on examples

        This method predicts based on a weighted combination of the true 
        target values of the examples. For a linear model, the predict 
        method has fitted coefficients Beta; example_predict uses the true 
        target values as the coefficients.

        Parameters
        ----------
        X : numpy.array
            (n observations x p features) matrix.

        Returns
        --------
        (n observations) numpy.array of predicted outputs.
        """
        return self.compute_weight(X) @ self.examples_y

    def compute_weight(self, X, examples_X=None):
        """Compute weight matrix W

        This method transforms a feature space X into a weight space W. The 
        weights W_ij is a logistic transformation of the (negative) distance 
        between observation i and example j.
        
        Parameters
        ----------
        X : numpy.array
            (n observations x p features) matrix.
        examples_X : numpy.array (optional)
            (m examples x p features) matrix. If None, use the examples_X 
            computed in the fit method.

        Returns
        -------
        (n x m) numpy.array where W_ij is the weight observation i attaches 
        to example j.
        """
        examples_X = self.examples_X if examples_X is None else examples_X
        dist = cdist(X, examples_X, *self.cdist_args, **self.cdist_kwargs)
        self.W = np.exp(-(dist+self.alpha))
        self.W /= self.W.sum(axis=1, keepdims=True)
        return self.W

    def fit_validate(self, X, y, max_examples=None):
        """Fit

        Parameters
        ----------
        X : numpy.array
            (n observations x p features) matrix.
        y : numpy.array
            (n observations) target array.
        max_examples : scalar (optional)
            Maximum number of examples to select.

        Returns
        -------
        Fitted model.
        """
        self.validating = True
        self.max_examples = max_examples or y.shape[0]
        self._train_test_split(X, y)
        self._compute_selected_idx()
        examples_X = np.take(self.X_train, self.selected_idx, axis=0)
        examples_y = np.take(self.y_train, self.selected_idx)
        del (
            self.validating, self.X_train, self.X_test, self.y_train, 
            self.y_test, self.W_train, self.W_test, self.selected_idx,
        )
        return self.fit(X, y, examples_X, examples_y)

    def _train_test_split(self, X, y):
        """Split into train and test (validation) data"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, *self.train_test_split_args, **self.train_test_split_kwargs
        )
        self.W_train = self.compute_weight(self.X_train, self.X_train)
        self.W_test = self.compute_weight(self.X_test, self.X_train)

    def _compute_selected_idx(self):
        """Stepwise selection of example observations"""
        # indices of selected examples from training data
        self.selected_idx = []
         # indices of candidate examples
        example_idx = list(range(self.W_train.shape[1]))
        # running best score
        best_score = -1
        # continue while test (validation) score shows improvement
        improvement = True
        while improvement and len(self.selected_idx) < self.max_examples:
            score = self._compute_score(example_idx)
            curr_best_score = score.max()
            improvement = curr_best_score > best_score
            if improvement:
                best_score = curr_best_score
                best_idx = score.argmax()
                self.selected_idx.append(best_idx)
                example_idx.remove(best_idx)

    def _compute_score(self, example_idx):
        """Compute score for all candidate examples"""
        score = np.zeros(self.W_train.shape[1])
        for idx in example_idx:
            selected_idx = self.selected_idx + [idx]
            y = np.take(self.y_train, selected_idx)

            # training
            W = self._preprocess(self.W_train, selected_idx)
            model = self.model.fit(W, self.y_train)

            # testing
            W = self._preprocess(self.W_test, selected_idx)
            score[idx] = model.score(W, self.y_test)
        return score

    def _preprocess(self, W, selected_idx):
        """Select examples from weight matrix and normalize"""
        W = np.take(W, selected_idx, axis=1)
        return W / W.sum(axis=1, keepdims=True)