import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin

class FeatureAlleleFrequencySelector(BaseEstimator, SelectorMixin):
    """Feature selector based on Allele Frequency. Features are selected on the basis of a threshold assigned for Frequency of A and Frequency of a. """

    @property
    def __name__(self):
        """Instance name is the same as the class name. """
        return self.__class__.__name__
    
    def __init__(self, threshold):
        """Create a FeatureAlleleFrequencySelector object.
        
        Parameters
        ----------
        threshold : float, required
            Threshold value for allele frequency. If frequency of A or frequency of a is less than the threshold value then the feature is dropped.
            
        Returns
        -------
        None
        
        """
        self.threshold = threshold

    def fit(self, X, y=None):
        """Fit FeatureAlleleFrequencySelector for feature selection
        
        Parameters
        ----------
        X : numpy ndarray, {n_samples, n_features}
            The training input samples.
        y : numpy array {n_samples,}
            The training target values.
            
        Returns
        -------
        self : object
            Returns a copy of the estimator
        """
        self.selected_feature_indexes = []
        self.no_of_features = X.shape[1]

        # Finding the no of alleles in each feature column
        for i in range(0, X.shape[1]):
            no_of_AA_featurewise = np.count_nonzero(X[:,i]==0)
            no_of_Aa_featurewise = np.count_nonzero(X[:,i]==1)
            no_of_aa_featurewise = np.count_nonzero(X[:,i]==2)
    
    
            frequency_A_featurewise = (2*no_of_AA_featurewise + no_of_Aa_featurewise) / (2*no_of_AA_featurewise + 
            2*no_of_Aa_featurewise + 2*no_of_aa_featurewise)

            frequency_a_featurewise = 1 - frequency_A_featurewise

            if(not(frequency_A_featurewise <= self.threshold) and not(frequency_a_featurewise <= self.threshold)):
                self.selected_feature_indexes.append(i)
        return self

    def transform(self, X):
        """Make subset after fit
        
        Parameters
        ----------
        X : numpy ndarray, {n_samples, n_features}
            New data, where n_samples is the number of samples and n_features is the number of features.
            
        Returns
        -------
        X_transformed : numpy ndarray, {n_samples, n_features}
            The transformed feature set.
        
        """
        X_transformed = X[:, self.selected_feature_indexes]

        return X_transformed

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected
        
        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for retention.
            """
        n_features = self.no_of_features
        mask = np.zeros(n_features, dtype=bool)
        mask[np.asarray(self.selected_feature_indexes)] = True

        return mask
