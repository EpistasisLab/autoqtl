

from sklearn.model_selection import train_test_split
from .base import AUTOQTLBase
from .config.regressor import regressor_config_dict

class AUTOQTLRegressor(AUTOQTLBase):
    """AUTOQTL estimator for regression problems. """
    scoring_function = 'r2' # Regression scoring
    default_config_dict = regressor_config_dict # Regression dictionary
    regression = True

    # Setting the sample of data used to verify pipelines work with the passed data set. Bypassing the _init_pretest() function for now.
    def _init_pretest(self, features, target):
        """Set the sample of data used to verify pipelines work with the passed data set.

        """
        self.pretest_X, _, self.pretest_y, _ = \
                train_test_split(
                                features,
                                target,
                                random_state=self.random_state,
                                test_size=None,
                                train_size=min(50,int(0.9*features.shape[0]))
                                )