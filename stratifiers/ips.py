from sklearn.model_selection._split import _BaseKFold
import numpy as np
from scipy.stats import wasserstein_distance


class IPSKFold(_BaseKFold):
    def __init__(self, n_splits=10, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        # Initializes equal split
        self.r = np.asarray([1 / self.n_splits] * self.n_splits)

    def split(self, dataset):
        self.dataset = dataset

        # Get dataaset information
        self.num_classes = self.dataset.num_classes
        self.num_samples = len(self.dataset)

        # Get mask information 
        self.pixel_counts = np.zeros([self.num_samples, self.num_classes])
        for i, (_, mask) in enumerate(self.dataset):
            self.pixel_counts[i,:] = [np.bincount(mask.flatten(), minlength=self.num_classes)[j] 
                                      for j in range(self.num_classes)]
        
        self.one_hot = np.zeros_like(self.pixel_counts)
        self.one_hot[self.pixel_counts != 0] = 1

        self.n_pixels_per_class =  self.pixel_counts.sum(axis = 0)

        self.desired_n_samples_in_fold = self.r * self.num_samples

        # Calculate optimal splits
        self.best_folds = self.optimize()

        # Calculate metrics for split
        self.edw, self.wd = self._fitness(self.best_folds, return_metrics=True)
        self.ed = np.mean(np.abs(self.n_samples_in_fold - self.desired_n_samples_in_fold))

        # Yield
        for fold_number in range(self.n_splits):
            test_indices = np.where(self.best_folds == fold_number)[0]
            train_indices = np.where(self.best_folds != fold_number)[0]
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    def optimize(self):
        best_individual = np.zeros(self.num_samples, dtype=int)
        desired_n_samples_in_folds = self.r * self.num_samples
        desired_n_pixels_in_folds_per_class = np.outer(self.r, self.n_pixels_per_class)
        samples_not_processed = np.ones(self.num_samples, dtype=bool)
        while np.any(samples_not_processed):
            num_pixels_for_classes = self.pixel_counts[samples_not_processed].sum(axis=0)
            class_idx_with_fewest_pixels = np.where(num_pixels_for_classes == num_pixels_for_classes[np.nonzero(num_pixels_for_classes)].min())[0]
            if class_idx_with_fewest_pixels.shape[0] > 1:    
                class_idx_with_fewest_pixels = class_idx_with_fewest_pixels[np.random.choice(class_idx_with_fewest_pixels.shape[0])] 
            sample_idxs = np.where(np.logical_and(self.one_hot[:, class_idx_with_fewest_pixels].flatten(), samples_not_processed))[0]
            for sample_idx in sample_idxs:
                class_folds = desired_n_pixels_in_folds_per_class[:, class_idx_with_fewest_pixels]
                fold_idx = np.where(class_folds == class_folds.max())[0]
                if fold_idx.shape[0] > 1:
                    temp_fold_idx = np.where(desired_n_samples_in_folds[fold_idx] ==
                                            desired_n_samples_in_folds[fold_idx].max())[0]
                    
                    if temp_fold_idx.shape[0] > 1:
                        fold_idx = fold_idx[temp_fold_idx]
                        fold_idx = fold_idx[np.random.choice(temp_fold_idx.shape[0])]
                    else:
                        fold_idx = fold_idx[temp_fold_idx[0]]
                else:
                    fold_idx = fold_idx[0]
                best_individual[sample_idx] = fold_idx
                samples_not_processed[sample_idx] = False
                desired_n_samples_in_folds[fold_idx] -= 1

                desired_n_pixels_in_folds_per_class[fold_idx][self.one_hot[sample_idx] == 1] -= self.pixel_counts[sample_idx][self.one_hot[sample_idx] == 1]

        return best_individual
    
    def _fitness(self, individual, return_metrics = False):

        self.n_samples_in_fold = np.bincount(individual, minlength=self.n_splits)

        self.n_pixels_in_fold_per_class = np.zeros((self.n_splits, self.num_classes))

        for fold_number in range(self.n_splits):
            samples_in_fold = [i == fold_number for i in individual]
            self.n_pixels_in_fold_per_class[fold_number, :] = np.sum(self.pixel_counts[samples_in_fold], axis=0)

        edw = self._ed_wasserstein()
        wd = self._wd()

        if return_metrics:
            return edw, wd
    
    def _ed_wasserstein(self):
        # get deviation from uniform distribution of samples/examples 
        edw = wasserstein_distance(np.arange(self.n_splits), 
                                                        np.arange(self.n_splits), 
                                                        self.n_samples_in_fold, 
                                                        self.r)
        
        return edw
    
    def _wd(self):
        pdm = []
        for fold_number in range(self.n_splits):
            pdm_f = wasserstein_distance(u_values= np.arange(self.num_classes).astype(np.float64), 
                                            v_values= np.arange(self.num_classes).astype(np.float64),
                                            u_weights= self.n_pixels_in_fold_per_class[fold_number].astype(np.float64), 
                                            v_weights= self.n_pixels_per_class.astype(np.float64))
            pdm.append(pdm_f)
        return np.mean(pdm)
    
# Example usage
if __name__ == "__main__":
    # from sklearn.datasets import make_classification
    # from sklearn.model_selection import cross_val_score
    # from sklearn.ensemble import RandomForestClassifier

    # X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    # custom_cv = CustomStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # model = RandomForestClassifier()

    # scores = cross_val_score(model, X, y, cv=custom_cv)
    # print("Cross-validation scores:", scores)

    mask = np.random.randint(0, 9, size=(1, 512, 512))
    # print([np.bincount(mask.flatten(), minlength=9)[j] for j in range(9)])


