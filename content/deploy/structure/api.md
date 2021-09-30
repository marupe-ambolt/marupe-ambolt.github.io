---
title: "API"
date: 2021-09-27T09:50:28+02:00
draft: false
weight: 2
---

The API is how users and other applications will interact with your service. We therefore need to specify how this interaction will happen. By default, an Emily machine learning API has three endpoints specified in the `api.py` file related to machine learning:

* `/api/train` endpoint
* `/api/evaluate` endpoint
* `/api/train` endpoint

We see in the notebook that the data is taken from an online repository:

```python
penguins = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")
```

This means that we can remove `dataset_path` as an input to the `/api/train/` and `/api/evaluate/` endpoints, since we will use this fixed dataset always:

```python
class TrainItem(BaseModel):
    save_path: str

@app.post('/api/train')
def train(item: TrainItem):
    return {'result': emily.train(item)}

class EvaluateItem(BaseModel):
    model_path: str

@app.post('/api/evaluate')
def evaluate(item: EvaluateItem):
    return {'result': emily.evaluate(item)}
```

Furthermore, if we inspect the code in the notebook, we can see that three features are used for predicting using regression:
* Flipper length
* Body mass
* Species

Because of this, we modify the `/api/predict/` endpoint to take these as input:

```python
class PredictItem(BaseModel):
    flipper_length: str
    body_mass: str
    species: str
    model_path: str

@app.post('/api/predict')
def predict(item: PredictItem):
    return {'result': emily.predict(item)}
```

### Full file

The `api.py` file now looks like this:

```python
import uvicorn
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from argparse import ArgumentParser

from utilities.utilities import get_uptime
from utilities.logging.config import initialize_logging, initialize_logging_middleware

from ml.emily import Emily

emily = Emily()

# --- Welcome to your Emily API! --- #
# See the README for guides on how to test it.

# Your API endpoints under http://yourdomain/api/...
# are accessible from any origin by default.
# Make sure to restrict access below to origins you
# trust before deploying your API to production.

parser = ArgumentParser()
parser.add_argument('-e', '--env', default='.env',
                    help='sets the environment file')
args = parser.parse_args()
dotenv_file = args.env
load_dotenv(dotenv_file)

app = FastAPI()

initialize_logging()
initialize_logging_middleware(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/api/health')
def health_check():
    return {
        'uptime': get_uptime(),
        'status': 'UP',
        'port': os.environ.get("HOST_PORT"),
    }


@app.get('/api')
def hello():
    return f'The API is running (uptime: {get_uptime()})'


class TrainItem(BaseModel):
    save_path: str


@app.post('/api/train')
def train(item: TrainItem):
    return {'result': emily.train(item)}


class EvaluateItem(BaseModel):
    model_path: str


@app.post('/api/evaluate')
def evaluate(item: EvaluateItem):
    return {'result': emily.evaluate(item)}


class PredictItem(BaseModel):
    flipper_length: str
    body_mass: str
    species: str
    model_path: str


@app.post('/api/predict')
def predict(item: PredictItem):
    return {'result': emily.predict(item)}


if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=os.environ.get('HOST_IP'),
        port=int(os.environ.get('CONTAINER_PORT'))
    )
```
