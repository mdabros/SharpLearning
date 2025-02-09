using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.GradientBoost.Test.Learners;

[TestClass]
public class RegressionQuantileLossGradientBoostLearnerTest
{
    [TestMethod]
    public void RegressionQuantileLossGradientBoostLearner_Learn()
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RegressionQuantileLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, 1.0, 0, 0.9, false);
        var model = sut.Learn(observations, targets);

        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.18540395091912656, actual);
    }

    [TestMethod]
    public void RegressionQuantileLossGradientBoostLearner_FeaturesPrSplit_Learn()
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RegressionQuantileLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, 1.0, 1, 0.9, false);
        var model = sut.Learn(observations, targets);

        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.96671358589437451, actual);
    }

    [TestMethod]
    public void RegressionQuantileLossGradientBoostLearner_Learn_Glass_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new RegressionQuantileLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 0, 0.9, false);

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

        Assert.AreEqual(1.1345507481360888, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionQuantileLossGradientBoostLearner_Learn_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new RegressionQuantileLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, 1.0, 0, 0.9, false);

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

        Assert.AreEqual(3.4664680050894057, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionQuantileLossGradientBoostLearner_Stochastic_Learn()
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RegressionQuantileLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 0, 0.9, false);
        var model = sut.Learn(observations, targets);

        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.10430373107075828, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionQuantileLossGradientBoostLearner_Stochastic_Learn_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new RegressionQuantileLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 0, 0.9, false);

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

        Assert.AreEqual(1.1345507481360888, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionQuantileLossGradientBoostLearner_Stochastic_FeaturesPrSplit_Learn_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new RegressionQuantileLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 3, 0.9, false);

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

        Assert.AreEqual(0.80797400652819307, actual, 0.0001);
    }
}
