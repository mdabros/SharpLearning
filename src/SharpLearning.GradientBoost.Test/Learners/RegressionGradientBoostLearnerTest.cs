﻿using System;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.GradientBoost.Test.Learners
{
    [TestClass]
    public class RegressionGradientBoostLearnerTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_Iterations()
        {
            new RegressionGradientBoostLearner(0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_LearningRate()
        {
            new RegressionGradientBoostLearner(1, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_MaximumTreeDepth()
        {
            new RegressionGradientBoostLearner(1, 1, -1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_MinimumSplitSize()
        {
            new RegressionGradientBoostLearner(1, 1, 1, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_MinimumInformationGain()
        {
            new RegressionGradientBoostLearner(1, 1, 1, 1, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_SubSampleRatio_TooLow()
        {
            new RegressionGradientBoostLearner(1, 1, 1, 1, 1, 0.0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_SubSampleRatio_TooHigh()
        {
            new RegressionGradientBoostLearner(1, 1, 1, 1, 1, 1.1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_FeaturesPrSplit()
        {
            new RegressionGradientBoostLearner(1, 1, 1, 1, 1, 1.0, -1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RegressionGradientBoostLearner_Constructor_Loss()
        {
            new RegressionGradientBoostLearner(1, 1, 1, 1, 1, 1.0, 1, null, false);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_LearnWithEarlyStopping_ToFewIterations()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();

            var splitter = new RandomTrainingTestIndexSplitter<double>(0.6, 1234);
            var split = splitter.SplitSet(observations, targets);

            var sut = new RegressionSquareLossGradientBoostLearner(5, 0.1, 3, 1, 1e-6, 1.0, 0, false);
            var evaluator = new MeanSquaredErrorRegressionMetric();

            var model = sut.LearnWithEarlyStopping(split.TrainingSet.Observations, split.TrainingSet.Targets,
                split.TestSet.Observations, split.TestSet.Targets, evaluator, 5);
        }

        [TestMethod]
        public void RegressionGradientBoostLearner_LearnWithEarlyStopping()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();

            var splitter = new RandomTrainingTestIndexSplitter<double>(0.6, 1234);
            var split = splitter.SplitSet(observations, targets);

            var sut = new RegressionSquareLossGradientBoostLearner(1000, 0.1, 3, 1, 1e-6, 1.0, 0, false);
            var evaluator = new MeanSquaredErrorRegressionMetric();

            var model = sut.LearnWithEarlyStopping(split.TrainingSet.Observations, split.TrainingSet.Targets,
                split.TestSet.Observations, split.TestSet.Targets, evaluator, 5);

            var predictions = model.Predict(split.TestSet.Observations);
            var actual = evaluator.Error(split.TestSet.Targets, predictions);

            Assert.AreEqual(0.061035472792879512, actual, 0.000001);
            Assert.AreEqual(40, model.Trees.Length);
        }

        [TestMethod]
        public void RegressionGradientBoostLearner_LearnWithEarlyStopping_array_not_long_enough()
        {
            var sut = new RegressionSquareLossGradientBoostLearner(
                iterations: 500,
                learningRate: 0.10936897046368986,
                maximumTreeDepth: 10,
                minimumSplitSize: 15,
                minimumInformationGain: 0.017451156205578751,
                subSampleRatio: 0.86159142598583882,
                featuresPrSplit: 144);

            IRegressionMetric metric = new MeanSquaredErrorRegressionMetric();

            var trainingRows = 5;
            var testRows = 9;
            var cols = 135;

            var split = new TrainingTestSetSplit(
                new F64Matrix(trainingRows, cols), Create1DArrayFilled(trainingRows, 1.0),
                new F64Matrix(testRows, cols), Create1DArrayFilled(testRows, 1.0));

            // Throws argument exception "Source array was not long enough"
            var model = sut.LearnWithEarlyStopping(
                split.TrainingSet.Observations, split.TrainingSet.Targets,
                split.TestSet.Observations, split.TestSet.Targets,
                metric,
                earlyStoppingRounds: 20);
        }

        static T[] Create1DArrayFilled<T>(int length, T fill)
        {
            var values = new T[length];
            Array.Fill(values, fill);
            return values;
        }
    }
}
