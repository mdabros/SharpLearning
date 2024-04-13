[![Build Status](https://github.com/mdabros/SharpLearning/actions/workflows/dotnet.yml/badge.svg?branch=master)](https://github.com/mdabros/SharpLearning/actions/workflows/dotnet.yml)
[![Nuget](https://img.shields.io/nuget/v/SharpLearning.Containers?color=purple)](https://www.nuget.org/packages/SharpLearning.Containers/)
[![downloads](https://img.shields.io/nuget/dt/SharpLearning.Containers)](https://www.nuget.org/packages/SharpLearning.Containers)
[![License](https://img.shields.io/github/license/mdabros/SharpLearning)](https://github.com/mdabros/SharpLearning/blob/master/LICENSE)

SharpLearning
=================

SharpLearning is an opensource machine learning library for C# .Net. 
The goal of SharpLearning is to provide .Net developers with easy access to machine learning algorithms and models.

Currently the main focus is supervised learning for classification and regression, 
while also providing the necesarry tools for optimizing and validating the trained models.

SharpLearning provides a simple high-level interface for machine learning algorithms.    
In SharpLearning a machine learning algorithm is refered to as a *Learner*, 
and a machine learning model is refered to as a *PredictorModel*. An example of usage can be seen below:

```c#
// Create a random forest learner for classification with 100 trees
var learner = new ClassificationRandomForestLearner(trees: 100);

// learn the model
var model = learner.Learn(observations, targets);

// use the model for predicting new observations
var predictions = model.Predict(testObservations);

// save the model for use with another application
model.Save(() => new StreamWriter("randomforest.xml"));
```

All machine learning algorithms and models implement the same interface for easy replacement.

Currently SharpLearning supports the following machine learning algorithms and models:

* DecisionTrees
* Adaboost (trees)
* GradientBoost (trees)
* RandomForest
* ExtraTrees
* NeuralNets (layers for fully connected and convolutional nets)
* Ensemble Learning

All the machine learning algorithms have sensible default hyperparameters for easy usage. 
However, several optimization methods are available for hyperparameter tuning:

* GridSearch
* RandomSearch
* ParticleSwarm
* GlobalizedBoundedNelderMead
* Hyperband
* BayesianOptimization  

License
-------

SharpLearning is covered under the terms of the [MIT](LICENSE) license. You may therefore link to it and use it in both opensource and proprietary software projects.

Documentation
-------------
SharpLearning contains xml documentation to help guide the user while using the library. 

Code examples and more information about how to use SharpLearning can be found in [SharpLearning.Examples](https://github.com/mdabros/SharpLearning.Examples)

The wiki also contains a set of guides on how to get started: 
 - [**Getting Started**](https://github.com/mdabros/SharpLearning/wiki/Getting-started).
 - [**Introduction to SharpLearning**](https://github.com/mdabros/SharpLearning/wiki/Introduction-to-SharpLearning).
 - [**Tuning Hyperparameters**](https://github.com/mdabros/SharpLearning/wiki/hyperparameter-tuning).
 - [**Using SharpLearning.XGBoost**](https://github.com/mdabros/SharpLearning/wiki/Using-SharpLearning.XGBoost)


Installation
------------

The recommended way to get SharpLearning is to use NuGet. The packages are provided and maintained in the public [NuGet Gallery](https://nuget.org/profiles/mdabros/).
More information can be found in the [getting started](https://github.com/mdabros/SharpLearning/wiki/Getting-started) guide on the wiki

Learner and model packages:

- **SharpLearning.DecisionTrees** - Provides learning algorithms and models for DecisionTree regression and classification.
- **SharpLearning.AdaBoost** - Provides learning algorithms and models for AdaBoost regression and classification.
- **SharpLearning.RandomForest** - Provides learning algorithms and models for RandomForest and ExtraTrees regression and classification.
- **SharpLearning.GradientBoost** - Provides learning algorithms and models for GradientBoost regression and classification.
- **SharpLearning.Neural** - Provides learning algorithms and models for neural net regression and classification. Layers available for fully connected and covolutional nets.
- **SharpLearning.XGBoost** - Provides learning algorithms and models for regression and classification using the [XGBoost library](https://github.com/dmlc/xgboost). CPU and GPU learning supported. This pakcage is x64 only.
- **SharpLearning.Ensemble** - Provides ensemble learning for regression and classification. Makes it possible to combine the other learners/models from SharpLearning.
- **SharpLearning.Common.Interfaces** - Provides common interfaces for SharpLearning.

Validation and model selection packages:

- **SharpLearning.CrossValidation** - Provides cross-validation, training/test set samplers and learning curves for SharpLearning.
- **SharpLearning.Metrics** - Provides classification, regression, impurity and ranking metrics..
- **SharpLearning.Optimization** - Provides optimization algorithms for hyperparameter tuning.

Container/IO packages:

- **SharpLearning.Containers** - Provides containers and base extension methods for SharpLearning.
- **SharpLearning.InputOutput** - Provides csv parsing and serialization for SharpLearning.
- **SharpLearning.FeatureTransformations** - Provides CsvRow transforms like missing value replacement and matrix transforms like MinMaxNormalization.

Contributing
------------
Contributions are welcome in the following areas:

 1. Add new issues with bug descriptions or feature suggestions.
 2. Add more examples to [SharpLearning.Examples](https://github.com/mdabros/SharpLearning.Examples).
 3. Solve existing issues by forking SharpLearning and creating a pull request.

When contributing, please follow the [contribution guide](https://github.com/mdabros/SharpLearning/blob/master/CONTRIBUTING.md).
