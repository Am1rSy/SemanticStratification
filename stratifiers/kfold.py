from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold
import numpy as np

class KFoldWrapper(KFold):
    def __init__(self, n_splits=10, shuffle=True, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        # Initializes equal split
    
    def split(self, dataset):
        self.dataset = dataset
        return super().split(range(len(dataset)))
    
