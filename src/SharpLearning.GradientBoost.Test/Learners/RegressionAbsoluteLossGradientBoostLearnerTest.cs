using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.GradientBoost.Test.Learners;

[TestClass]
public class RegressionAbsoluteLossGradientBoostLearnerTest
{
    [TestMethod]
    public void RegressionAbsoluteLossGradientBoostLearner_Learn()
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RegressionAbsoluteLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, 1.0, 0, false);
        var model = sut.Learn(observations, targets);

        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.03309315166467057, actual);
    }

    [TestMethod]
    public void RegressionAbsoluteLossGradientBoostLearner_FeaturesPrSplit_Learn()
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RegressionAbsoluteLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, 1.0, 1, false);
        var model = sut.Learn(observations, targets);

        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.0861480348494789, actual);
    }

    [TestMethod]
    public void RegressionAbsoluteLossGradientBoostLearner_Learn_Glass_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new RegressionAbsoluteLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 0, false);

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

        Assert.AreEqual(0.34134304639324115, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionAbsoluteLossGradientBoostLearner_Learn_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new RegressionAbsoluteLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, 1.0, 0, false);

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

        Assert.AreEqual(0.41246374405350877, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionAbsoluteLossGradientBoostLearner_Stochastic_Learn()
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RegressionAbsoluteLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 0, false);
        var model = sut.Learn(observations, targets);

        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.033412842952357739, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionAbsoluteLossGradientBoostLearner_Stochastic_Learn_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new RegressionAbsoluteLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 0, false);

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

        Assert.AreEqual(0.34134304639324115, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionAbsoluteLossGradientBoostLearner_Stochastic_FeaturesPrSplit_Learn_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new RegressionAbsoluteLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 3, false);

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

        Assert.AreEqual(0.38258161801010859, actual, 0.0001);
    }
}
