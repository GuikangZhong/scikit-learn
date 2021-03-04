import pytest
import numpy as np
import re
from sklearn.impute import SimpleImputer

@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_simple_imputation_None_missing_value_error_handling(strategy):
    arr = np.array([[0], [1], [2], [None]])

    imputer = SimpleImputer(missing_values=None, strategy=strategy)
    err_msg = "Cannot use {} strategy with non-numeric data:\n" \
    "Input contains NaN, infinity or a value too large for dtype(\'float64\').\n" \
    "To treat None as np.NaN, recommend using \"missing_values=np.NaN\" " \
    "when instantiate the imputer".format(strategy)
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        imputer.fit_transform(arr)