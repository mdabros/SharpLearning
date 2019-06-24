﻿using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Ensemble.Test.Learners
{
    [TestClass]
    public class ClassificationEnsembleLearnerTest
    {
        [TestMethod]
        public void ClassificationEnsembleLearner_Learn()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var sut = new ClassificationEnsembleLearner(learners, 
                new MeanProbabilityClassificationEnsembleStrategy());

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.14485981308411214, actual, 0.0001);
        }

        [TestMethod]
        public void ClassificationEnsembleLearner_Learn_Bagging()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var sut = new ClassificationEnsembleLearner(learners, 
                new MeanProbabilityClassificationEnsembleStrategy(), 0.7);

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.16822429906542055, actual, 0.0001);
        }

        [TestMethod]
        public void ClassificationEnsembleLearner_Learn_Indexed()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var sut = new ClassificationEnsembleLearner(learners, 
                new MeanProbabilityClassificationEnsembleStrategy());

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var indices = Enumerable.Range(0, 25).ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.67289719626168221, actual, 0.0001);
        }
    }
}
