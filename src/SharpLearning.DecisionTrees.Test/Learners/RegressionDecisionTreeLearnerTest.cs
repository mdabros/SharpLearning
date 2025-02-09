using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.DecisionTrees.Test.Learners;

[TestClass]
public class RegressionDecisionTreeLearnerTest
{
    [TestMethod]
    public void RegressionDecisionTreeLearner_Learn_Reuse_No_Valid_Split()
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RegressionDecisionTreeLearner();

        // train initial model.
        sut.Learn(observations, targets);

        // reuse learner, with smaller data that provides no valid split.
        var onlyUniqueTargetValue = 1.0;
        var onlyOneUniqueObservations = (F64Matrix)observations.Rows(0, 1, 2, 3, 4);
        var onlyOneUniquetargets = Enumerable.Range(0, onlyOneUniqueObservations.RowCount).Select(v => onlyUniqueTargetValue).ToArray();
        var model = sut.Learn(onlyOneUniqueObservations, onlyOneUniquetargets);

        var predictions = model.Predict(onlyOneUniqueObservations);
        // no valid split, so should result in the model always returning the onlyUniqueTargetValue.
        for (var i = 0; i < predictions.Length; i++)
        {
            Assert.AreEqual(onlyUniqueTargetValue, predictions[i], 0.0001);
        }
    }

    [TestMethod]
    public void RegressionDecisionTreeLearner_Learn_Depth_100()
    {
        var error = RegressionDecisionTreeLearner_Learn(100);
        Assert.AreEqual(0.032120286249559482, error, 0.0000001);
    }

    [TestMethod]
    public void RegressionDecisionTreeLearner_Learn_Depth_1()
    {
        var error = RegressionDecisionTreeLearner_Learn(1);
        Assert.AreEqual(0.55139468272009107, error, 0.0000001);
    }

    [TestMethod]
    public void RegressionDecisionTreeLearner_Learn_Depth_3()
    {
        var error = RegressionDecisionTreeLearner_Learn(2);
        Assert.AreEqual(0.14322350107327153, error, 0.0000001);
    }

    [TestMethod]
    public void RegressionDecisionTreeLearner_Learn_Depth_100_Weight_1()
    {
        var error = RegressionDecisionTreeLearner_Learn_Weighted(100, 1.0);
        Assert.AreEqual(0.032120286249559482, error, 0.0000001);
    }

    [TestMethod]
    public void RegressionDecisionTreeLearner_Learn_Depth_1_Weight_1()
    {
        var error = RegressionDecisionTreeLearner_Learn_Weighted(1, 1);
        Assert.AreEqual(0.55139468272009107, error, 0.0000001);
    }

    [TestMethod]
    public void RegressionDecisionTreeLearner_Learn_Depth_3_Weight_1()
    {
        var error = RegressionDecisionTreeLearner_Learn_Weighted(2, 1);
        Assert.AreEqual(0.14322350107327153, error, 0.0000001);
    }

    [TestMethod]
    public void RegressionDecisionTreeLearner_Learn_Depth_100_Weight_100()
    {
        var error = RegressionDecisionTreeLearner_Learn_Weighted(100, 100.0);
        Assert.AreEqual(0.032256921590414704, error, 0.0000001);
    }

    static double RegressionDecisionTreeLearner_Learn(int treeDepth)
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RegressionDecisionTreeLearner(treeDepth, 4, 2, 0.1, 42);

        var model = sut.Learn(observations, targets);

        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var error = evaluator.Error(targets, predictions);
        return error;
    }

    double RegressionDecisionTreeLearner_Learn_Weighted(int treeDepth, double weight)
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RegressionDecisionTreeLearner(treeDepth, 4, 2, 0.1, 42);
        var weights = targets.Select(v => Weight(v, weight)).ToArray();
        var model = sut.Learn(observations, targets, weights);

        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var error = evaluator.Error(targets, predictions);
        return error;
    }

    static double Weight(double v, double weight)
    {
        if (v < 3.0)
            return weight;
        return 1.0;
    }
}
