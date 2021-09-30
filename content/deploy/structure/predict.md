---
title: "Predicting"
date: 2021-09-27T09:50:28+02:00
draft: false
weight: 6
---

Now that everything else has been set up, we are ready for doing predictions. As a first step, we import the numpy library, which we will use later.

```python
import numpy as np
from ml.model import Model
```

### *_preprocess* method

The `_preprocess()` method should take as input the features we use to predict.
* Flipper length
* Body mass
* Species

Then the method creates an array and puts the sample input into that array, and finally standardises it using the same scaler as previously, to make sure that the sample is scaled in the same way as the training data.

```python
def _preprocess(self, flipper_length, body_mass, species):
    array = np.zeros((1,5), dtype = int)
    array[0][0] = flipper_length
    array[0][1] = body_mass
    array[0][2] = 0
    array[0][3] = 0
    array[0][4] = 0
    if (species == 'Adelie'):
        array[0][2] = 1
    if (species == 'Chinstrap'):
        array[0][3] = 1
    if (species == 'Gentoo'):
        array[0][4]

    array[0,0:2] = self.model.scaler.transform(array[:,0:2])

    return array
```

### *_postprocess* method
For the postprocessing step, we simply add a bit of explanatory text to the result:

```python
def _postprocess(self, prediction):
    return "Predicted culmen length is " + str(prediction[0]) + " mm"
```

### *train* method
For the actual training, all we need to do is to make sure to unpack each of the features in the sample, as well as modify the code slightly to fit the changes made to the other methods:

```python
def predict(self, request):
    """
    Performs prediction on a sample using the model at the given path
    """

    # Unpack request
    flipper_length = request.flipper_length
    body_mass = request.body_mass
    species = request.species
    model_path = request.model_path

    # Loads a trained instance of the Model class
    # If no model has been trained yet proceed to follow the steps in ml.trainer.py
    if model_path != self.model_path:
        self.model = Model()
        self.model.load_model(model_path)
        self.model_path = model_path

    # Preprocess the inputted sample to prepare it for the model
    preprocessed_sample = self._preprocess(flipper_length, body_mass, species)

    # Forward the preprocessed sample into the model as defined in the __call__ function in the Model class
    prediction = self.model(preprocessed_sample)

    # Postprocess the prediction to prepare it for the client
    prediction = self._postprocess(prediction)

    return prediction
```

### Full file

The `predictory.py` file now looks like this:

```python
import numpy as np
from ml.model import Model


class Predictor:
    """
    The Predictor class is used for making predictions using a trained model instance based on the Model class
    defined in ml.model.py and the training steps defined in ml.trainer.py
    """

    def __init__(self):
        self.model_path = ""
        self.model = None

    def predict(self, request):
        """
        Performs prediction on a sample using the model at the given path
        """

        # Unpack request
        flipper_length = request.flipper_length
        body_mass = request.body_mass
        species = request.species
        model_path = request.model_path

        # Loads a trained instance of the Model class
        # If no model has been trained yet proceed to follow the steps in ml.trainer.py
        if model_path != self.model_path:
            self.model = Model()
            self.model.load_model(model_path)
            self.model_path = model_path

        # Preprocess the inputted sample to prepare it for the model
        preprocessed_sample = self._preprocess(flipper_length, body_mass, species)

        # Forward the preprocessed sample into the model as defined in the __call__ function in the Model class
        prediction = self.model(preprocessed_sample)

        # Postprocess the prediction to prepare it for the client
        prediction = self._postprocess(prediction)

        return prediction

    def _preprocess(self, flipper_length, body_mass, species):
        array = np.zeros((1,5), dtype = int)
        array[0][0] = flipper_length
        array[0][1] = body_mass
        array[0][2] = 0
        array[0][3] = 0
        array[0][4] = 0
        if (species == 'Adelie'):
            array[0][2] = 1
        if (species == 'Chinstrap'):
            array[0][3] = 1
        if (species == 'Gentoo'):
            array[0][4]

        array[0,0:2] = self.model.scaler.transform(array[:,0:2])

        return array

    def _postprocess(self, prediction):
        return "Predicted culmen length is " + str(prediction[0]) + " mm"

    def __call__(self, request):
        return self.predict(request)
```
