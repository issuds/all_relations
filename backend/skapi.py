"""
Scikit - learn based api for estimation of all 1 -> 1 or n -> 1
relations in the dataset.
"""

import numpy as np
import pandas as ps
from tqdm import tqdm
#import bootstrapped.bootstrap as bs
#import bootstrapped.stats_functions as bs_stats

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import r2_score

from sklearn.preprocessing import RobustScaler, StandardScaler, Imputer
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict

from skopt import gp_minimize

from string import digits

def pandas_to_concepts(data):
    """
    Converts pandas dataframe to set of all concepts.

    Parameters
    ----------
    data: DataFrame, contains the dataset

    Returns
    -------
    result: dict with all concepts
    """

    result = {}

    if not isinstance(data, ps.DataFrame):
        raise ValueError("Parameter data should be of type DataFrame.")

    for c in data.columns:
        x = data[c]

        # select only the name of concept here
        concept = c
        while concept[-1] in digits:
            concept = concept[:-1]

        if not concept in result:
            result[concept] = np.array(x)[:, np.newaxis]
        else:
            result[concept] = np.column_stack([result[concept], x])

    return result


def make_regressor(model_subset=None):
    """
    Generate the necessary estimator model class for grid search.

    Parameters
    ----------

    * model_subset [string, default=None]
        Whether to use a named model_subset of model classes for fitting.
        Feasible options are:
        - None: use all the models available
        - 'linear': use only linear models

    Returns
    -------
    model: GridSearchCV instance, estimator class that can be applied
        to features to learn the relationship.
    """
    estimator = Pipeline([
        ('imputer', Imputer()),
        ('scale', StandardScaler()),
        ('model', GradientBoostingRegressor()),
    ])

    # search spaces for different model classes
    model_choices = {
        "lasso": {
            'model': [Lasso()],
            'model__alpha': [10 ** i for i in np.linspace(-6, 6, 11)],
        },
        "ann": {
            'model': [MLPRegressor(solver='lbfgs')],
            'model__hidden_layer_sizes': [[n_neurons for _ in range(n_layers)] for n_neurons in [4, 16, 64, 256] for n_layers in [1, 2]]
        },
        "knn": {
            'model': [KNeighborsRegressor()],
            'model__n_neighbors': range(1, 100, 5),
        },
        "gbrt": {
            'model': [GradientBoostingRegressor()],
            'model__n_estimators': [2 ** i for i in range(1, 9)],
            'model__learning_rate': [2 ** i for i in range(-10, 0)],
        }
    }

    dctree = {
        'model': [DecisionTreeRegressor()],
        'model__max_depth': range(1, 20),
        'model__min_samples_split': [2 ** i for i in range(-20, -1)],
    }
    
    # user can specify subset of models to be used
    if model_subset is None:
        choices = model_choices.keys()
    else:
        choices = model_subset

    spaces = [model_choices[k] for k in choices]

    # this class search over all parameter spaces for parameter
    # combination which yields the best validation loss
    model = GridSearchCV(
        estimator=estimator,
        param_grid=spaces, # knn, gbrt, dectree
        n_jobs=-1,
        verbose=1,
    )

    return model


def mapping_power(X, Y, models_subset=None):
    """
    Evaluate the strength of relation from X to Y.

    Parameters
    ----------

    * models_subset [string, default=None]
        See same argument of the make_regressor function.
    * X [np.ndarray, shape=(n_samples, n_features)]
        Array of input concept observations. Missing values
        are denoted with nan's.

    Returns
    -------
    model: GridSearchCV instance, estimator class that can be applied
        to features to learn the relationship.
    """
    # evaluate all the models in cross - validation fashion
    y_true, y_pred = [], []

    # iterate over all columns
    for y in Y.T:
        I = ~np.isnan(y) # select rows where outputs are not missing

        yp = cross_val_predict(make_regressor(models_subset), X[I], y[I])

        y_true.append(y[I])
        y_pred.append(yp)

    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)

    # calculate bootstrap on rmsea
    #n_iter, p = 1000000, 0.000001
    #print(bs.bootstrap((yt-yp)**2, stat_func=bs_stats.mean, alpha=p, num_iterations=n_iter, iteration_batch_size=10000))
    #print(bs.bootstrap((yt-np.mean(yt))**2, stat_func=bs_stats.mean, alpha=p, num_iterations=n_iter, iteration_batch_size=10000))

    # compare the cross - validation predictions for all columns
    score = r2_score(
        yt,
        yp
    )

    return score


def all_1_to_1(concepts, prefix = None, models_subset = None):
    """
    Finds all one to one relations within the set of concepts.

    Parameters
    ----------
    concepts : dict, where every element is a numpy array of shape
        [n_samples, n_features']. Training data, where n_samples in
        the number of samples of records describing particular
        concept. n_features' can be different for different concepts.

    prefix : array-like, shape = [n_samples, n_features]
        Features that apply to every concept.

    models_subset: string or None
        Whether to use a subset of models for estimation of mapping
        power. For feasible options, see the similar parameter of
        the `mapping_power` function.


    Returns
    -------
    result : array of [set a, set b, float]
        Returns estimate of test accuracy for how single concept in
        set b can be estimated from single concept in set a. All
        combinations of single concepts are considered. The value
        of float represents how well the concept can be estimated.

    """

    names = concepts.keys()
    result = []

    for A in tqdm(names):
        for B in names:

            if A == B:
                continue

            print(A, "->", B)

            X = concepts[A]

            if prefix is not None:
                X = np.column_stack([prefix, X])

            Y = concepts[B]
            Y = Y.astype('float')

            local_result = [[A], [B]]

            # get score for estimation of non - missing values
            score = mapping_power(X, Y, models_subset=models_subset)
            local_result.append(score)

            result.append(local_result)

    return result


def concept_subset(concepts, names, prefix = None):
    selection = [concepts[n] for n in names]

    if prefix is not None:
        selection += prefix

    result = np.column_stack(selection)
    return result




def all_n_to_1(concepts, prefix = None, discount=0.95, max_iter=32):
    """
    Finds all one to one relations within the set of concepts.

    Parameters
    ----------
    concepts : dict, where every element is a numpy array of shape
        [n_samples, n_features']. Training data, where n_samples in
        the number of samples of records describing particular
        concept. n_features' can be different for different concepts.

    prefix : array-like, shape = [n_samples, n_features]
        Features that apply to every concept.

    discount : float
        Fraction of r^2 with all concepts that should be preserved
        with the subset of concepts.

    max_iter : int
        Number of iterations used in black box optimization algorithm.


    Returns
    -------
    result : array of [set a, set b, float]
        Returns estimate of test accuracy for how single concept in
        set b can be estimated from minimum set of all concepts (set a)
        such that the total accuracy is not worse than discount*100%
        compared to when all concepts are used.
        The value of float represents how well the concept can be estimated -
        0.0 is worst, 1.0 is best.

    """

    names = set(concepts.keys())
    result = []

    for B in tqdm(names):
        # first try with all concepts
        inputs = np.array([v for v in (names - {B})])
        X = concept_subset(concepts, inputs, prefix)
        Y = concept_subset(concepts, {B})

        # initial score
        baseline = mapping_power(X, Y)

        if baseline < 0.0:
            result.append([{}, {B}, 0.0])
            continue

        # objective: minimize number of input nodes, while
        # maintaining fraction of baseline performance
        space = [(True, False) for v in inputs]
        space_length = len(space)*1.0

        pbar = tqdm(total=max_iter)
        current_iter = [0]

        def obj(selection):
            selection = np.array(selection)

            if np.all(~selection):
                return 1.0

            selection = inputs[selection]

            X = concept_subset(concepts, selection, prefix)
            Y = concept_subset(concepts, {B})

            performance = mapping_power(X, Y)

            pbar.update(1)

            if performance < baseline * discount:
                return 1.0
            else:
                return len(selection) / space_length

        solution = gp_minimize(obj, space, n_calls=max_iter)
        found_concepts = inputs[np.array(solution.x)]
        found_concepts = set(found_concepts)

        result.append([list(found_concepts), [B], baseline*discount])

    return result
