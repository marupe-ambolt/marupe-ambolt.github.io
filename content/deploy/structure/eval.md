---
title: "Evaluating"
date: 2021-09-27T09:50:28+02:00
draft: false
weight: 5
---

In the `evaluator.py` file, we can also remove mentions of the `dataset_path` variable. Many of the steps for evaluating are the same as for training, so we import the same libraries:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import Model
```

### *_load_test_data* method

In order to load the test data, we do essentially the same as when loading the train data, except now we return the test data instead of the train data.

```python
def _load_test_data(self):
    data = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")
    return data
```

### *_preprocess_test_data* method

Likewise for the preprocessing step, we mirror the steps taken for the train data

```python
def _preprocess_test_data(self, data):
    data.dropna(inplace=True)

    y = data['culmen_length_mm']
    X_dum = data.species_short.str.get_dummies()
    X = pd.concat([data.iloc[:,4:6], X_dum], axis=1)
    X.iloc[:,0:2] = self.model.scaler.transform(X.iloc[:,0:2])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)        
    return X_test, y_test
```

Note the `random_state=21` parameter, which ensures that the data is split the same way each time.

### *evaluate* method

First we modify the existing code to match the changes made to the `_load_test_data()` and `_preprocess_test_data` methods:

```python
def evaluate(self, request):
    """
    Evaluates a trained model located at 'model_path' based on test data from the self._load_test_data function
    """

    # Unpack request
    model_path = request.model_path

    # Loads a trained instance of the Model class
    # If no model has been trained yet proceed to follow the steps in ml.trainer.py
    if model_path != self.model_path:
        self.model = Model()
        self.model.load_model(model_path)
        self.model_path = model_path

    # Read the dataset from the dataset_path
    data = self._load_test_data()

    # Preprocess dataset to prepare it for the evaluator
    X_test, y_test = self._preprocess_test_data(data)
```

We see in the notebook that the model is evaluated like this:

```python
model_ols.score(X_test, y_test)
```

Putting this into the `train()` method we obtain:

```python
score = self.model.score(X_test, y_test)

return score
```

### Full file

The `evaluator.py` file now looks like this:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import Model


class Evaluator:
    """
    The Evaluator class is used for evaluating a trained model instance.
    In order to get started with evaluating a model the following steps needs to be taken:
    1. Train a model following the steps in ml.trainer.py
    2. Prepare test data on which the model should be evaluated on by implementing the _read_test_data() function and
    the _preprocess_test_data function
    """

    def __init__(self):
        self.model_path = ""
        self.model = None

    def evaluate(self, request):
        """
        Evaluates a trained model located at 'model_path' based on test data from the self._load_test_data function
        """

        # Unpack request
        model_path = request.model_path

        # Loads a trained instance of the Model class
        # If no model has been trained yet proceed to follow the steps in ml.trainer.py
        if model_path != self.model_path:
            self.model = Model()
            self.model.load_model(model_path)
            self.model_path = model_path

        # Read the dataset from the dataset_path
        data = self._load_test_data()

        # Preprocess dataset to prepare it for the evaluator
        X_test, y_test = self._preprocess_test_data(data)

        score = self.model.score(X_test, y_test)

        return score

    def _load_test_data(self):
        data = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")
        return data

    def _preprocess_test_data(self, data):
        data.dropna(inplace=True)

        y = data['culmen_length_mm']
        X_dum = data.species_short.str.get_dummies()
        X = pd.concat([data.iloc[:,4:6], X_dum], axis=1)
        X.iloc[:,0:2] = self.model.scaler.transform(X.iloc[:,0:2])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)        
        return X_test, y_test

    def __call__(self, request):
        return self.evaluate(request)
```
