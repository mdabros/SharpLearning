using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.GradientBoost.Test.Learners;

[TestClass]
public class RegressionSquareLossGradientBoostLearnerTest
{
    [TestMethod]
    public void RegressionSquareLossGradientBoostLearner_Learn()
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RegressionSquareLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, 1.0, 0, false);
        var model = sut.Learn(observations, targets);

        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.023436850973295304, actual);
    }

    [TestMethod]
    public void RegressionSquareLossGradientBoostLearner_FeaturesPrSplit_Learn()
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RegressionSquareLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, 1.0, 1, false);
        var model = sut.Learn(observations, targets);

        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.074376126071145687, actual);
    }

    [TestMethod]
    public void RegressionSquareLossGradientBoostLearner_Learn_Glass_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new RegressionSquareLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 0, false);

        var indices = Enumerable.Range(0, targets.Length).ToArray();
        indices.Shuffle(new Random(42));
        indices = indices.Take((int)(targets.Length * 0.7))
            .ToArray();

        var model = sut.Learn(observations, targets, indices);
        var predictions = model.Predict(observations);
        var indexedPredictions = predictions.GetIndices(indices);
        var indexedTargets = targets.GetIndices(indices);

        var evaluator = new MeanAbsolutErrorRegressionMetric();
        var actual = evaluator.Error(indexedTargets, indexedPredictions);

        Assert.AreEqual(0.29330180231264918, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionSquareLossGradientBoostLearner_Learn_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new RegressionSquareLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, 1.0, 0, false);

        var indices = Enumerable.Range(0, targets.Length).ToArray();
        indices.Shuffle(new Random(42));
        indices = indices.Take((int)(targets.Length * 0.7))
            .ToArray();

        var model = sut.Learn(observations, targets, indices);
        var predictions = model.Predict(observations);
        var indexedPredictions = predictions.GetIndices(indices);
        var indexedTargets = targets.GetIndices(indices);

        var evaluator = new MeanAbsolutErrorRegressionMetric();
        var actual = evaluator.Error(indexedTargets, indexedPredictions);

        Assert.AreEqual(0.23625469946001074, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionSquareLossGradientBoostLearner_Stochastic_Learn()
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RegressionSquareLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 0, false);
        var model = sut.Learn(observations, targets);

        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.025391913155163696, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionSquareLossGradientBoostLearner_Stochastic_Learn_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new RegressionSquareLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 0, false);

        var indices = Enumerable.Range(0, targets.Length).ToArray();
        indices.Shuffle(new Random(42));
        indices = indices.Take((int)(targets.Length * 0.7))
            .ToArray();

        var model = sut.Learn(observations, targets, indices);
        var predictions = model.Predict(observations);
        var indexedPredictions = predictions.GetIndices(indices);
        var indexedTargets = targets.GetIndices(indices);

        var evaluator = new MeanAbsolutErrorRegressionMetric();
        var actual = evaluator.Error(indexedTargets, indexedPredictions);

        Assert.AreEqual(0.29330180231264918, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionSquareLossGradientBoostLearner_Stochastic_FeaturesPrSplit_Learn_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new RegressionSquareLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 3, false);

        var indices = Enumerable.Range(0, targets.Length).ToArray();
        indices.Shuffle(new Random(42));
        indices = indices.Take((int)(targets.Length * 0.7))
            .ToArray();

        var model = sut.Learn(observations, targets, indices);
        var predictions = model.Predict(observations);
        var indexedPredictions = predictions.GetIndices(indices);
        var indexedTargets = targets.GetIndices(indices);

        var evaluator = new MeanAbsolutErrorRegressionMetric();
        var actual = evaluator.Error(indexedTargets, indexedPredictions);

        Assert.AreEqual(0.32442703567005193, actual, 0.0001);
    }
}
