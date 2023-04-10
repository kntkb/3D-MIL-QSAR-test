import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from typing import Sequence, Tuple, Union
import logging


# Settings
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class RandomForestRegressorWrapper():
    """ Random forest regressor.
    """
    def __init__(self, search_method, params, n_cv=3, n_iter=100, random_seed=1):
        self.search_method = search_method
        self.params = params
        self.n_cv = n_cv
        self.n_iter = n_iter
        self.random_seed = random_seed
        self.rf = RandomForestRegressor()   

    def fit(self, x: Union[Sequence[Union[Sequence, np.array]], np.array], y: Union[Sequence,np.array]):
        """ Fit model.

        Use the random grid or grid search to search for best hyperparameters before fitting.
        First create the base model to tune.
        """
        # hypertuning
        if self.search_method == "random":
            search_model = RandomizedSearchCV(estimator=self.rf, param_distributions=self.params, n_iter=self.n_iter, cv=self.n_cv, n_jobs=-1, random_state=self.random_seed, verbose=2)
        elif self.search_method == "grid":
            search_model = GridSearchCV(estimator=self.rf, param_grid=self.params, cv=self.n_cv, n_jobs=-1, verbose=2)

        search_model.fit(x, y)
        logging.debug(f"Best model parameters: {search_model.best_params_}")
        self.best_model = search_model.best_estimator_
        return self

    def predict(self, x):
        y = self.best_model.predict(x)
        return np.array(y).flatten()


class RandomForestClassifierWrapper(RandomForestRegressorWrapper):
    """ Random forest classifier.
    """
    def __init__(self, search_method, params, n_cv=3, n_iter=100, random_seed=1):
        super().__init__(search_method, params, n_cv=3, n_iter=100, random_seed=1)