using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.DecisionTrees.Test.suts;

[TestClass]
public class ClassificationDecisionTreeModelTest
{
    readonly string m_classificationDecisionTreeModelString = "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<ClassificationDecisionTreeModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Models\">\r\n  <Tree xmlns:d2p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\" z:Id=\"2\">\r\n    <d2p1:Nodes z:Id=\"3\" z:Size=\"5\">\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>0</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>4</d2p1:RightIndex>\r\n        <d2p1:Value>20</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>-1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>2</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>1</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>3</d2p1:RightIndex>\r\n        <d2p1:Value>2.5</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>0</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>2</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>1</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>1</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>3</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>0</d2p1:Value>\r\n      </d2p1:Node>\r\n      <d2p1:Node>\r\n        <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n        <d2p1:LeafProbabilityIndex>2</d2p1:LeafProbabilityIndex>\r\n        <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n        <d2p1:NodeIndex>4</d2p1:NodeIndex>\r\n        <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n        <d2p1:Value>1</d2p1:Value>\r\n      </d2p1:Node>\r\n    </d2p1:Nodes>\r\n    <d2p1:Probabilities xmlns:d3p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"4\" z:Size=\"3\">\r\n      <d3p1:ArrayOfdouble z:Id=\"5\" z:Size=\"2\">\r\n        <d3p1:double>0.3333333333333333</d3p1:double>\r\n        <d3p1:double>0.6666666666666666</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n      <d3p1:ArrayOfdouble z:Id=\"6\" z:Size=\"2\">\r\n        <d3p1:double>0.7391304347826086</d3p1:double>\r\n        <d3p1:double>0.2608695652173913</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n      <d3p1:ArrayOfdouble z:Id=\"7\" z:Size=\"2\">\r\n        <d3p1:double>0.16666666666666666</d3p1:double>\r\n        <d3p1:double>0.8333333333333333</d3p1:double>\r\n      </d3p1:ArrayOfdouble>\r\n    </d2p1:Probabilities>\r\n    <d2p1:TargetNames xmlns:d3p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"8\" z:Size=\"2\">\r\n      <d3p1:double>0</d3p1:double>\r\n      <d3p1:double>1</d3p1:double>\r\n    </d2p1:TargetNames>\r\n    <d2p1:VariableImportance xmlns:d3p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"9\" z:Size=\"2\">\r\n      <d3p1:double>0</d3p1:double>\r\n      <d3p1:double>0.1803324880247957</d3p1:double>\r\n    </d2p1:VariableImportance>\r\n  </Tree>\r\n  <m_variableImportance xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"9\" i:nil=\"true\" />\r\n</ClassificationDecisionTreeModel>";

    [TestMethod]
    public void ClassificationDecisionTreeModel_Predict_Single()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new ClassificationDecisionTreeLearner(100, 1, 2, 0.001, 42);
        var sut = learner.Learn(observations, targets);

        var rows = targets.Length;
        var predictions = new double[rows];
        for (int i = 0; i < rows; i++)
        {
            predictions[i] = sut.Predict(observations.Row(i));
        }

        var evaluator = new TotalErrorClassificationMetric<double>();
        var error = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.038461538461538464, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeModel_Precit_Multiple()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new ClassificationDecisionTreeLearner(100, 1, 2, 0.001, 42);
        var sut = learner.Learn(observations, targets);

        var predictions = sut.Predict(observations);

        var evaluator = new TotalErrorClassificationMetric<double>();
        var error = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.038461538461538464, error, 0.0000001);
    }

    [TestMethod]
    public void ClassificationDecisionTreeModel_Predict_Multiple_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new ClassificationDecisionTreeLearner(100, 5, 2, 0.001, 42);
        var sut = learner.Learn(observations, targets);

        var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };
        var predictions = sut.Predict(observations, indices);

        var evaluator = new TotalErrorClassificationMetric<double>();
        var indexedTargets = targets.GetIndices(indices);
        var error = evaluator.Error(indexedTargets, predictions);

        Assert.AreEqual(0.1, error, 0.0000001);
    }


    [TestMethod]
    public void ClassificationDecisionTreeModel_PredictProbability_Single()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new ClassificationDecisionTreeLearner(100, 5, 2, 0.001, 42);
        var sut = learner.Learn(observations, targets);

        var rows = targets.Length;
        var actual = new ProbabilityPrediction[rows];
        for (int i = 0; i < rows; i++)
        {
            actual[i] = sut.PredictProbability(observations.Row(i));
        }

        var evaluator = new TotalErrorClassificationMetric<double>();
        var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

        Assert.AreEqual(0.23076923076923078, error, 0.0000001);

        var expected = new ProbabilityPrediction[] { new(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), };
        CollectionAssert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void ClassificationDecisionTreeModel_PredictProbability_Multiple()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new ClassificationDecisionTreeLearner(100, 5, 2, 0.001, 42);
        var sut = learner.Learn(observations, targets);

        var actual = sut.PredictProbability(observations);
        var evaluator = new TotalErrorClassificationMetric<double>();
        var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

        Assert.AreEqual(0.23076923076923078, error, 0.0000001);

        var expected = new ProbabilityPrediction[] { new(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), };
        CollectionAssert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void ClassificationDecisionTreeModel_PredictProbability_Multiple_Indexed()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new ClassificationDecisionTreeLearner(100, 5, 2, 0.001, 42);
        var sut = learner.Learn(observations, targets);

        var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };
        var actual = sut.PredictProbability(observations, indices);

        var indexedTargets = targets.GetIndices(indices);
        var evaluator = new TotalErrorClassificationMetric<double>();
        var error = evaluator.Error(indexedTargets, actual.Select(p => p.Prediction).ToArray());

        Assert.AreEqual(0.1, error, 0.0000001);

        var expected = new ProbabilityPrediction[] { new(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new(0, new Dictionary<double, double> { { 0, 0.571428571428571 }, { 1, 0.428571428571429 }, }), new(1, new Dictionary<double, double> { { 0, 0.428571428571429 }, { 1, 0.571428571428571 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new(0, new Dictionary<double, double> { { 0, 0.75 }, { 1, 0.25 }, }), new(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), new(0, new Dictionary<double, double> { { 0, 0.857142857142857 }, { 1, 0.142857142857143 }, }), new(1, new Dictionary<double, double> { { 0, 0.285714285714286 }, { 1, 0.714285714285714 }, }), };
        CollectionAssert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void ClassificationDecisionTreeModel_GetVariableImportance()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, { "PreviousExperience_month", 1 } };

        var learner = new ClassificationDecisionTreeLearner(100, 1, 2, 0.001, 42);
        var sut = learner.Learn(observations, targets);

        var actual = sut.GetVariableImportance(featureNameToIndex);
        var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, { "AptitudeTestScore", 19.5121951219512 } };

        Assert.AreEqual(expected.Count, actual.Count);
        var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

        foreach (var item in zip)
        {
            Assert.AreEqual(item.Expected.Key, item.Actual.Key);
            Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
        }
    }

    [TestMethod]
    public void ClassificationDecisionTreeModel_GetRawVariableImportance()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new ClassificationDecisionTreeLearner(100, 1, 2, 0.001, 42);
        var sut = learner.Learn(observations, targets);

        var actual = sut.GetRawVariableImportance();
        var expected = new double[] { 0.071005917159763288, 0.36390532544378695 };

        Assert.AreEqual(expected.Length, actual.Length);

        for (int i = 0; i < expected.Length; i++)
        {
            Assert.AreEqual(expected[i], actual[i], 0.000001);
        }
    }

    [TestMethod]
    public void ClassificationDecisionTreeModel_Save()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new ClassificationDecisionTreeLearner(2);
        var sut = learner.Learn(observations, targets);

        var writer = new StringWriter();
        sut.Save(() => writer);

        var actual = writer.ToString();
        Assert.AreEqual(m_classificationDecisionTreeModelString, actual);
    }

    [TestMethod]
    public void ClassificationDecisionTreeModel_Load()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var reader = new StringReader(m_classificationDecisionTreeModelString);
        var sut = ClassificationDecisionTreeModel.Load(() => reader);

        var predictions = sut.Predict(observations);

        var evaluator = new TotalErrorClassificationMetric<double>();
        var error = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.19230769230769232, error, 0.0000001);
    }

    static void Write(ProbabilityPrediction[] predictions)
    {
        var value = "new ProbabilityPrediction[] {";
        foreach (var item in predictions)
        {
            value += "new ProbabilityPrediction(" + item.Prediction + ", new Dictionary<double, double> {";
            foreach (var prob in item.Probabilities)
            {
                value += "{" + prob.Key + ", " + prob.Value + "}, ";
            }
            value += "}),";
        }
        value += "};";

        Trace.WriteLine(value);
    }
}
