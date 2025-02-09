using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.DecisionTrees.Test.Learners;

[TestClass]
public class ClassificationDecisionTreeLearnerTest
{
    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Reuse_No_Valid_Split()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new ClassificationDecisionTreeLearner();

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
    public void ClassificationDecisionTreeLearner_Learn_Aptitude_Depth_100()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Aptitude(100);
        Assert.AreEqual(0.038461538461538464, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Aptitude_depth_1()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Aptitude(1);
        Assert.AreEqual(0.23076923076923078, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Aptitude_depth_5()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Aptitude(5);
        Assert.AreEqual(0.076923076923076927, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Glass_100()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Glass(100);
        Assert.AreEqual(0.0, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Glass_Depth_1()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Glass(1);
        Assert.AreEqual(0.5280373831775701, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Glass_Depth_5()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Glass(5);
        Assert.AreEqual(0.16355140186915887, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Aptitude_Depth_100_Weight_1()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Aptitude_Weighted(100, 1);
        Assert.AreEqual(0.038461538461538464, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Aptitude_depth_1_Weight_1()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Aptitude_Weighted(1, 1);
        Assert.AreEqual(0.23076923076923078, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Aptitude_depth_5_Weight_1()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Aptitude_Weighted(5, 1);
        Assert.AreEqual(0.076923076923076927, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Glass_100_Weight_1()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Glass_Weighted(100, 1);
        Assert.AreEqual(0.0, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Glass_Depth_1_Weight_1()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Glass_Weighted(1, 1);
        Assert.AreEqual(0.5280373831775701, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Glass_Depth_5_Weight_1()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Glass_Weighted(5, 1);
        Assert.AreEqual(0.16355140186915887, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Aptitude_Depth_100_Weight_10()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Aptitude_Weighted(100, 10);
        Assert.AreEqual(0.076923076923076927, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Aptitude_depth_5_Weight_10()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Aptitude_Weighted(5, 10);
        Assert.AreEqual(0.076923076923076927, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Glass_100_Weight_10()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Glass_Weighted(100, 10);
        Assert.AreEqual(0.070093457943925228, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeLearner_Learn_Glass_Depth_5_Weight_10()
    {
        var error = ClassificationDecisionTreeLearner_Learn_Glass_Weighted(5, 10);
        Assert.AreEqual(0.14018691588785046, error, 0.0000001);
    }

    static double ClassificationDecisionTreeLearner_Learn_Glass(int treeDepth)
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var sut = new ClassificationDecisionTreeLearner(treeDepth, 1, observations.ColumnCount, 0.001, 42);
        var model = sut.Learn(observations, targets);

        var predictions = model.Predict(observations);

        var evaluator = new TotalErrorClassificationMetric<double>();
        var error = evaluator.Error(targets, predictions);
        return error;
    }

    static double ClassificationDecisionTreeLearner_Learn_Aptitude(int treeDepth)
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var sut = new ClassificationDecisionTreeLearner(treeDepth, 1, 2, 0.001, 42);
        var model = sut.Learn(observations, targets);

        var predictions = model.Predict(observations);

        var evaluator = new TotalErrorClassificationMetric<double>();
        var error = evaluator.Error(targets, predictions);
        return error;
    }

    double ClassificationDecisionTreeLearner_Learn_Glass_Weighted(int treeDepth, double weight)
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var weights = targets.Select(v => Weight(v, 1, weight)).ToArray();
        var sut = new ClassificationDecisionTreeLearner(treeDepth, 1, observations.ColumnCount, 0.001, 42);
        var model = sut.Learn(observations, targets, weights);

        var predictions = model.Predict(observations);
        var evaluator = new TotalErrorClassificationMetric<double>();
        Trace.WriteLine(evaluator.ErrorString(targets, predictions));
        var error = evaluator.Error(targets, predictions);
        return error;
    }

    double ClassificationDecisionTreeLearner_Learn_Aptitude_Weighted(int treeDepth, double weight)
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var weights = targets.Select(v => Weight(v, 0, weight)).ToArray();
        var sut = new ClassificationDecisionTreeLearner(treeDepth, 1, 2, 0.001, 42);
        var model = sut.Learn(observations, targets, weights);

        var predictions = model.Predict(observations);

        var evaluator = new TotalErrorClassificationMetric<double>();
        Trace.WriteLine(evaluator.ErrorString(targets, predictions));
        var error = evaluator.Error(targets, predictions);
        return error;
    }

    public static double Weight(double v, double targetToWeigh, double weight)
    {
        if (v == targetToWeigh)
            return weight;
        return 1.0;
    }
}
