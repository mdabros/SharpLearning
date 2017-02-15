SharpLearning
=================

SharpLearning is an opensource machine learning library for C# .Net. 
The goal of SharpLearning is to provide .Net developers with easy access to machine learning algorithms and models.

Currently the main focus is supervised learning using classification and regression, 
while also providing the necesarry tools for optimizing and validating the trained models.

SharpLearning provides a unified interface for machine learning algorithms. In SharpLearning a machine learning algorithm is refered to as a *Learner*, 
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
All machine learning algorithms and models use the same interface for easy replacement.

Currently SharpLearning supports the following machine learning algorithms and models:

* DecisionTrees
* Adaboost (trees)
* GradientBoost (trees)
* RandomForest
* ExtraTrees
* NeuralNets (layers for fully connected and convolutional nets)
* Ensemble Learning

All the machine learning algorithms have sensible default hyperparameters for easy usage. 
However, several optimization methods are availible for hyperparameter tuning:

* GridSearch
* RandomSearch
* ParticleSwarm
* GlobalizedBoundedNelderMead
* SequentialModelBased  

License
-------

SharpLearning is covered under the terms of the [MIT](LICENSE.md) license. You may therefore link to it and use it in both opensource and proprietary software projects.

Documentation
-------------
The code contains xml comments to help guide users. 

A separate repository with examples is planned as well as further markdown dokumentation.

Installation
------------

The recommended way to get SharpLearning is to use NuGet. The packages are provided and maintained in the public [NuGet Gallery](https://nuget.org/profiles/mdabros/).

The nuget packages will be availible shortly.

