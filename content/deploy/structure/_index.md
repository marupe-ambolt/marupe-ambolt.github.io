---
title: "Code structure"
date: 2021-09-27T09:50:28+02:00
draft: false
weight: 1
---

Many AI and ML projects start as Jupyter notebooks, where one can easily explore data as well as try and compare different models. However, once the project has matured enough that a model has been selected, one should think about how to structure the code in a way that makes it easy to deploy as a service.

We will make use of the notebook
https://colab.research.google.com/drive/1QktyJ61oU926-zRcX_zoli6knSYUZBu8
where, among other things, regression is used on a penguin dataset. We will identify the relevant parts of this regression and put them into an Emily API project. An Emily API project can be created by running `emily build` and choosing the Machine learning API template.
