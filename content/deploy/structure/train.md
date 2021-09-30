---
title: "Training"
date: 2021-09-27T09:50:28+02:00
draft: false
weight: 4
---

In the `trainer.py` file, we can do away with any mention of the `dataset_path` variable, since we removed this from the `/api/train/` endpoint.

### *_load_train_data* method

We first implement the `_load_train_data()` method, which, as the name suggests, loads the training data.
To do this, we first look at the notebook and see that the data is read from a github page using the pandas library:

```python
penguins = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")
```

Therefore we need to import the pandas library, and we will also import the `train_test_split` library from scikitlearn:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import Model
```

Now we can use pandas to read the data:

```python
def _load_train_data(self):
    data = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")
    return data
```

### *_preprocess_train_data* method

We see in the following lines of code from the notebook how to split the data into feature variables (`X`) and target variable (`y`):

```python
penguins.dropna(inplace=True)
```

```python
y = penguins['culmen_length_mm']
```

```python
X_dum = penguins.species_short.str.get_dummies()
```

```python
X = pd.concat([penguins.iloc[:,4:6], X_dum], axis=1)
```

```python
X.iloc[:,0:2] = StandardScaler().fit_transform(X.iloc[:,0:2])
```

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)
```

We put these parts of the notebook into the `_preprocess_train_data()` method:

```python
def _preprocess_train_data(self, data):
    data.dropna(inplace=True)        

    y = data['culmen_length_mm']
    X_dum = data.species_short.str.get_dummies()
    X = pd.concat([data.iloc[:,4:6], X_dum], axis=1)
    X.iloc[:,0:2] = self.model.scaler.fit_transform(X.iloc[:,0:2])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    return X_train, y_train
```

### *train* method
First we modify the existing code so it conforms to the modifications we have made in the previous methods:

```python
def train(self, request):
    """
    Starts the training of a model based on data loaded by the self._load_train_data function
    """

    # Unpack request
    save_path = request.save_path

    # Read the dataset from the dataset_path
    data = self._load_train_data()

    # Preprocess the dataset
    X_train, y_train = self._preprocess_train_data(data)
```

The relevant code for actually training the model in the notebook is:

```python
model_ols.fit(X_train, y_train)
```

We duplicate this in the `train()` method:

```python
self.model.fit(X_train, y_train)

# Save the trained model
return self.model.save_model(save_path)
```

### Full file

The `trainer.py` file now looks like this:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import Model


class Trainer:
    """
    The Trainer class is used for training a model instance based on the Model class found in ml.model.py.
    In order to get started with training a model the following steps needs to be taken:
    1. Define the Model class in ml.model.py
    2. Prepare train data on which the model should be trained with by implementing the _read_train_data() function and
    the _preprocess_train_data() function
    """

    def __init__(self):
        self.model = Model()  # creates an instance of the Model class (see guidelines in ml.model.py)

    def train(self, request):
        """
        Starts the training of a model based on data loaded by the self._load_train_data function
        """

        # Unpack request
        save_path = request.save_path

        # Read the dataset from the dataset_path
        data = self._load_train_data()

        # Preprocess the dataset
        X_train, y_train = self._preprocess_train_data(data)

        self.model.fit(X_train, y_train)

        # Save the trained model
        return self.model.save_model(save_path)

    def _load_train_data(self):
        data = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")
        return data

    def _preprocess_train_data(self, data):
        data.dropna(inplace=True)        

        y = data['culmen_length_mm']
        X_dum = data.species_short.str.get_dummies()
        X = pd.concat([data.iloc[:,4:6], X_dum], axis=1)
        X.iloc[:,0:2] = self.model.scaler.fit_transform(X.iloc[:,0:2])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
        return X_train, y_train

    def __call__(self, request):
        return self.train(request)
```
