---
title: "Model"
date: 2021-09-27T09:50:28+02:00
draft: false
weight: 3
---

In the notebook we see the following snippet of code where a linear regression model is defined.

```python
model_ols = LinearRegression()
```

In the `model.py` file, we therefore specify that our model should be a `LinearRegression` model:

```python
from sklearn.linear_model import LinearRegression

class Model(LinearRegression):
```

We also add a scaler to the model, which we will use to standardise data at various stages:

```python 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
```

```python
def __init__(self):
    super().__init__()  # Inherit methods from the super class which this class extends from
    self.scaler = StandardScaler()
```

### *forward* method

When forwarding a sample to the model, it should simply predict using the `predict()` method:

```python
def forward(self, sample):
  return self.predict(sample)
```

### *save_model* and *load_model* methods

Finally, to save and load a model, we use the `pickle` library, which must be imported:

```python
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
```

Now we can implement saving and loading:

```python
    def save_model(self, save_path):
        with open(save_path, 'wb') as fp:
            pickle.dump(self, fp)

    def load_model(self, model_path):
        with open(model_path, 'rb') as fp:
            model = pickle.load(fp)
            self.__dict__.update(model.__dict__)
```

### Full file

The `model.py` file now looks like this:
```python
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

class Model(LinearRegression):

    def __init__(self):
        super().__init__()  # Inherit methods from the super class which this class extends from
        self.scaler = StandardScaler()

    def forward(self, sample):
        return self.predict(sample)

    def save_model(self, save_path):
        with open(save_path, 'wb') as fp:
            pickle.dump(self, fp)

    def load_model(self, model_path):
        with open(model_path, 'rb') as fp:
            model = pickle.load(fp)
            self.__dict__.update(model.__dict__)

    def __call__(self, sample):
        return self.forward(sample)
``` 
