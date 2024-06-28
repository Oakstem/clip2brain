import numpy as np
from sklearn.model_selection import BaseCrossValidator


class RDMCrossValidator(BaseCrossValidator):
    def __init__(self, n_splits: int = 5, groups: np.array = None):
        assert groups.shape[1] == 2

        self.groups = groups
        self.unique_items = list(set(np.unique(groups[:, 0])) | set(np.unique(groups[:, 1])))
        self.unique_items = np.array(self.unique_items)
        # shuffle the unique items:
        np.random.shuffle(self.unique_items)
        # split to n_splits:
        self.test_splits = np.array_split(self.unique_items, n_splits)
        self.train_splits = [np.concatenate([split for j, split in enumerate(self.test_splits) if j != i]) for i in range(n_splits)]

        self.n_splits = n_splits

    def _extract_indices(self, indices):
        return np.where(np.isin(self.groups[:, 0], indices) & np.isin(self.groups[:, 1], indices))[0]

    def split(self, X, y, groups=None):
        for i in range(self.n_splits):
            train_indices = self._extract_indices(self.train_splits[i])
            test_indices = self._extract_indices(self.test_splits[i])
            yield train_indices, test_indices

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

def manual_cv(X, y, kfold, groups=5):
    kfold.get_n_splits(X, y, groups)

    print("GroupKFold:")
    for train_index, test_index in kfold.split(X, y, groups):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("X_train:", X_train, "X_test:", X_test)
        print("y_train:", y_train, "y_test:", y_test)