_link to [car data.csv](https://github.com/marsianjohncarter/StackOverflow--Sklearn_ValueError)_



My code:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
car_data = pd.read_csv('car_data.csv')

# Create X
X = car_data.drop('Buy Rate', axis=1)

# Create Y
y = car_data['Buy Rate']

clf = RandomForestClassifier()
clf.get_params()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)
```

After the line with `clf.fit`, this error pops up:

```
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_51905/2395142735.py in ?()
----> 1 clf.fit(X_train, y_train)

~/Desktop/ml-course/env/lib/python3.10/site-packages/sklearn/base.py in ?(estimator, *args, **kwargs)
   1147                 skip_parameter_validation=(
   1148                     prefer_skip_nested_validation or global_skip_validation
   1149                 )
   1150             ):
-> 1151                 return fit_method(estimator, *args, **kwargs)

~/Desktop/ml-course/env/lib/python3.10/site-packages/sklearn/ensemble/_forest.py in ?(self, X, y, sample_weight)
    344         """
    345         # Validate or convert input data
    346         if issparse(y):
    347             raise ValueError("sparse multilabel-indicator for y is not supported.")
--> 348         X, y = self._validate_data(
    349             X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
    350         )
    351         if sample_weight is not None:

~/Desktop/ml-course/env/lib/python3.10/site-packages/sklearn/base.py in ?(self, X, y, reset, validate_separately, cast_to_ndarray, **check_params)
    617                 if "estimator" not in check_y_params:
    618                     check_y_params = {**default_check_params, **check_y_params}
    619                 y = check_array(y, input_name="y", **check_y_params)
    620             else:
--> 621                 X, y = check_X_y(X, y, **check_params)
    622             out = X, y
    623 
    624         if not no_val_X and check_params.get("ensure_2d", True):

~/Desktop/ml-course/env/lib/python3.10/site-packages/sklearn/utils/validation.py in ?(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1143         raise ValueError(
   1144             f"{estimator_name} requires y to be passed, but the target y is None"
   1145         )
   1146 
-> 1147     X = check_array(
   1148         X,
   1149         accept_sparse=accept_sparse,
   1150         accept_large_sparse=accept_large_sparse,

~/Desktop/ml-course/env/lib/python3.10/site-packages/sklearn/utils/validation.py in ?(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    914                         )
    915                     array = xp.astype(array, dtype, copy=False)
    916                 else:
    917                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
--> 918             except ComplexWarning as complex_warning:
    919                 raise ValueError(
    920                     "Complex data not supported\n{}\n".format(array)
    921                 ) from complex_warning

~/Desktop/ml-course/env/lib/python3.10/site-packages/sklearn/utils/_array_api.py in ?(array, dtype, order, copy, xp)
    376         # Use NumPy API to support order
    377         if copy is True:
    378             array = numpy.array(array, order=order, dtype=dtype)
    379         else:
--> 380             array = numpy.asarray(array, order=order, dtype=dtype)
    381 
    382         # At this point array is a NumPy ndarray. We convert it to an array
    383         # container that is consistent with the input's namespace.

~/Desktop/ml-course/env/lib/python3.10/site-packages/pandas/core/generic.py in ?(self, dtype)
   2082     def __array__(self, dtype: npt.DTypeLike | None = None) -> np.ndarray:
   2083         values = self._values
-> 2084         arr = np.asarray(values, dtype=dtype)
   2085         if (
   2086             astype_is_view(values.dtype, arr.dtype)
   2087             and using_copy_on_write()

ValueError: could not convert string to float: 'Hyundai'
```
