using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Metrics.Regression;
using SharpLearning.XGBoost.Learners;
using SharpLearning.XGBoost.Models;

namespace SharpLearning.XGBoost.Test.Learners;

[TestClass]
public class RegressionXGBoostModelTest
{
    readonly double m_delta = 0.0000001;

    [TestMethod]
    public void RegressionXGBoostModel_Predict_Single()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var learner = CreateLearner();

        using var sut = learner.Learn(observations, targets);
        var rows = targets.Length;
        var predictions = new double[rows];
        for (var i = 0; i < rows; i++)
        {
            predictions[i] = sut.Predict(observations.Row(i));
        }

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var error = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.0795934933096642, error, m_delta);
    }

    [TestMethod]
    public void RegressionXGBoostModel_Predict_Multiple()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var learner = CreateLearner();

        using var sut = learner.Learn(observations, targets);
        var predictions = sut.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var error = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.0795934933096642, error, m_delta);
    }

    [TestMethod]
    public void RegressionXGBoostModel_Save_Load()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var learner = CreateLearner();
        var sut = learner.Learn(observations, targets);

        var predictions = sut.Predict(observations);
        var modelFilePath = "model.xgb";

        using (var sutPreSave = learner.Learn(observations, targets))
        {
            AssertModel(observations, targets, sutPreSave);
            sutPreSave.Save(modelFilePath);
        }

        using var sutAfterSave = RegressionXGBoostModel.Load(modelFilePath);
        AssertModel(observations, targets, sutAfterSave);
    }

    [TestMethod]
    public void RegressionXGBoostModel_GetVariableImportance()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var index = 0;
        var name = "f";
        var featureNameToIndex = Enumerable.Range(0, 9)
            .ToDictionary(v => name + index.ToString(), v => index++);

        var learner = CreateLearner();

        using var sut = learner.Learn(observations, targets);
        var actual = sut.GetVariableImportance(featureNameToIndex);
        var expected = new Dictionary<string, double>
            {
                { "f2", 100 },
                { "f7", 21.1439170859871 },
                { "f6", 17.5087210061721 },
                { "f3", 12.6909395202158 },
                { "f0", 12.3235851417467 },
                { "f1", 9.00304680229703 },
                { "f5", 7.10296482157573 },
                { "f4", 6.43327754840246 },
                { "f8", 4.61553313147666 },
            };

        Assert.AreEqual(expected.Count, actual.Count);
        var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

        foreach (var item in zip)
        {
            Assert.AreEqual(item.Expected.Key, item.Actual.Key);
            Assert.AreEqual(item.Expected.Value, item.Actual.Value, m_delta);
        }
    }

    static RegressionXGBoostLearner CreateLearner()
    {
        return new RegressionXGBoostLearner(maximumTreeDepth: 3,
            learningRate: 0.1,
            estimators: 100,
            silent: true,
            objective: RegressionObjective.LinearRegression,
            boosterType: BoosterType.GBTree,
            treeMethod: TreeMethod.Auto,
            numberOfThreads: -1,
            gamma: 0,
            minChildWeight: 1,
            maxDeltaStep: 0,
            subSample: 1,
            colSampleByTree: 1,
            colSampleByLevel: 1,
            l1Regularization: 0,
            l2Reguralization: 1,
            scalePosWeight: 1,
            baseScore: 0.5,
            seed: 0,
            missing: double.NaN);
    }

    void AssertModel(F64Matrix observations, double[] targets, RegressionXGBoostModel model)
    {
        var predictions = model.Predict(observations);
        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.0795934933096642, actual, m_delta);
    }
}
