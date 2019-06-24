﻿using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TimeSeries;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.CrossValidation.Test.TimeSeries
{
    [TestClass]
    public class TimeSeriesCrossValidationTest
    {
        [TestMethod]
        public void TimeSeriesCrossValidation_Validate()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var sut = new TimeSeriesCrossValidation<double>(initialTrainingSize: 5);

            var learner = new RegressionDecisionTreeLearner();
            var timeSeriesPredictions = sut.Validate(learner, observations, targets);
            var timeSeriesTargets = sut.GetValidationTargets(targets);

            var metric = new MeanSquaredErrorRegressionMetric();
            var error = metric.Error(timeSeriesTargets, timeSeriesPredictions);

            Assert.AreEqual(0.098690664447830825, error, 0.00001);
        }
        
        [TestMethod]
        public void TimeSeriesCrossValidation_Validate_MaxTrainingSetSize()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var sut = new TimeSeriesCrossValidation<double>(initialTrainingSize: 5, maxTrainingSetSize: 10);

            var learner = new RegressionDecisionTreeLearner();
            var timeSeriesPredictions = sut.Validate(learner, observations, targets);
            var timeSeriesTargets = sut.GetValidationTargets(targets);

            var metric = new MeanSquaredErrorRegressionMetric();
            var error = metric.Error(timeSeriesTargets, timeSeriesPredictions);

            Assert.AreEqual(0.27296549371962692, error, 0.00001);
        }

        [TestMethod]
        public void TimeSeriesCrossValidation_Validate_RetrainInterval()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var sut = new TimeSeriesCrossValidation<double>(initialTrainingSize: 5, retrainInterval: 5);

            var learner = new RegressionDecisionTreeLearner();
            var timeSeriesPredictions = sut.Validate(learner, observations, targets);
            var timeSeriesTargets = sut.GetValidationTargets(targets);

            var metric = new MeanSquaredErrorRegressionMetric();
            var error = metric.Error(timeSeriesTargets, timeSeriesPredictions);

            Assert.AreEqual(0.096346937132994928, error, 0.00001);
        }

        [TestMethod]
        public void TimeSeriesCrossValidation_Validate_MaxTrainingSetSize_And_RetrainInterval()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var sut = new TimeSeriesCrossValidation<double>(initialTrainingSize: 5, maxTrainingSetSize: 30, retrainInterval: 5);

            var learner = new RegressionDecisionTreeLearner();
            var timeSeriesPredictions = sut.Validate(learner, observations, targets);
            var timeSeriesTargets = sut.GetValidationTargets(targets);

            var metric = new MeanSquaredErrorRegressionMetric();
            var error = metric.Error(timeSeriesTargets, timeSeriesPredictions);

            Assert.AreEqual(0.13010151998135897, error, 0.00001);
        }

        [TestMethod]
        public void TimeSeriesCrossValidation_GetValidationTargets()
        {
            var random = new Random(23);
            var targets = Enumerable.Range(0, 100).Select(v => (double)random.Next()).ToArray();
            var initialTrainingSize = 5;

            var sut = new TimeSeriesCrossValidation<double>(initialTrainingSize);

            var actual = sut.GetValidationTargets(targets);
            var expected = targets.Skip(initialTrainingSize).ToArray();

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void TimeSeriesCrossValidation_GetValidationIndices()
        {
            var targets = Enumerable.Range(0, 100).Select(v => (double)v).ToArray();
            var initialTrainingSize = 5;

            var sut = new TimeSeriesCrossValidation<double>(initialTrainingSize);

            var actual = sut.GetValidationIndices(targets);
            var expected = targets.Skip(initialTrainingSize).Select(v => (int)v).ToArray();

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TimeSeriesCrossValidation_Validate_Observations_And_Targets_Length_Does_Not_Match()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();
            targets = targets.Take(100).ToArray();

            var sut = new TimeSeriesCrossValidation<double>(initialTrainingSize: 5);

            var learner = new RegressionDecisionTreeLearner();
            var timeSeriesPredictions = sut.Validate(learner, observations, targets);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TimeSeriesCrossValidation_Validate_InitialTrainingSize_Is_Larger_Than_Obsevations_Length()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var sut = new TimeSeriesCrossValidation<double>(initialTrainingSize: 300);

            var learner = new RegressionDecisionTreeLearner();
            var timeSeriesPredictions = sut.Validate(learner, observations, targets);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TimeSeriesCrossValidation_InitialTrainingSize_Is_Zero()
        {
            new TimeSeriesCrossValidation<double>(initialTrainingSize: 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TimeSeriesCrossValidation_MaxTrainingSetSize_Is_Smaller_Than_Zero()
        {
            new TimeSeriesCrossValidation<double>(initialTrainingSize: 5, maxTrainingSetSize: -1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TimeSeriesCrossValidation_InitialTrainingSize_Is_Larger_Than_MaxTrainingSetSize()
        {
            new TimeSeriesCrossValidation<double>(initialTrainingSize: 5, maxTrainingSetSize: 4);
        }
    }
}
