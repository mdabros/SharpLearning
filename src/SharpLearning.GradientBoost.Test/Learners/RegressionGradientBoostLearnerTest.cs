using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.GradientBoost.Test.Learners;

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
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

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
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

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
    [ExpectedException(typeof(InvalidOperationException))]
    public void RegressionGradientBoostLearner_LearnWithEarlyStopping_when_more_featuresPerSlit_than_featureCount_Throw()
    {
        var sut = new RegressionSquareLossGradientBoostLearner(500, 0.1, 10, 15, 0.01, 0.8,
            featuresPrSplit: 4);

        IRegressionMetric metric = new MeanSquaredErrorRegressionMetric();

        var trainingRows = 5;
        var testRows = 6;
        var cols = 3;

        var split = new TrainingTestSetSplit(
            new F64Matrix(trainingRows, cols), new double[trainingRows],
            new F64Matrix(testRows, cols), new double[testRows]);

        var model = sut.LearnWithEarlyStopping(
            split.TrainingSet.Observations, split.TrainingSet.Targets,
            split.TestSet.Observations, split.TestSet.Targets,
            metric,
            earlyStoppingRounds: 20);
    }
}
