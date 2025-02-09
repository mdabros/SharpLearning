using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Metrics.Regression;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Models;

namespace SharpLearning.RandomForest.Test.Models;

[TestClass]
public class RegressionForestModelTest
{
    readonly double m_delta = 0.0000001;
    readonly string m_regressionForestModelString =
"<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<RegressionForestModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.RandomForest.Models\">\r\n  <_x003C_Trees_x003E_k__BackingField xmlns:d2p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Models\" z:Id=\"2\" z:Size=\"2\">\r\n    <d2p1:RegressionDecisionTreeModel z:Id=\"3\">\r\n      <d2p1:Tree xmlns:d4p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\" z:Id=\"4\">\r\n        <d4p1:Nodes z:Id=\"5\" z:Size=\"7\">\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>0</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>0</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>4</d4p1:RightIndex>\r\n            <d4p1:Value>3.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>2</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>1</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>3</d4p1:RightIndex>\r\n            <d4p1:Value>13.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>0</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>2</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>3</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0.42857142857142855</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>5</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>4</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>6</d4p1:RightIndex>\r\n            <d4p1:Value>17</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>2</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>5</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0.6</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>3</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>6</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0.5</d4p1:Value>\r\n          </d4p1:Node>\r\n        </d4p1:Nodes>\r\n        <d4p1:Probabilities xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"6\" z:Size=\"4\">\r\n          <d5p1:ArrayOfdouble z:Id=\"7\" z:Size=\"0\" />\r\n          <d5p1:ArrayOfdouble z:Id=\"8\" z:Size=\"0\" />\r\n          <d5p1:ArrayOfdouble z:Id=\"9\" z:Size=\"0\" />\r\n          <d5p1:ArrayOfdouble z:Id=\"10\" z:Size=\"0\" />\r\n        </d4p1:Probabilities>\r\n        <d4p1:TargetNames xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"11\" z:Size=\"2\">\r\n          <d5p1:double>0</d5p1:double>\r\n          <d5p1:double>1</d5p1:double>\r\n        </d4p1:TargetNames>\r\n        <d4p1:VariableImportance xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"12\" z:Size=\"2\">\r\n          <d5p1:double>0.757342657342657</d5p1:double>\r\n          <d5p1:double>0.40714285714285714</d5p1:double>\r\n        </d4p1:VariableImportance>\r\n      </d2p1:Tree>\r\n      <d2p1:m_variableImportance xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"12\" i:nil=\"true\" />\r\n    </d2p1:RegressionDecisionTreeModel>\r\n    <d2p1:RegressionDecisionTreeModel z:Id=\"13\">\r\n      <d2p1:Tree xmlns:d4p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\" z:Id=\"14\">\r\n        <d4p1:Nodes z:Id=\"15\" z:Size=\"7\">\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>0</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>6</d4p1:RightIndex>\r\n            <d4p1:Value>20</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>2</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>1</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>3</d4p1:RightIndex>\r\n            <d4p1:Value>9.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>0</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>2</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0.1111111111111111</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>0</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>4</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>3</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>5</d4p1:RightIndex>\r\n            <d4p1:Value>2</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>4</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0.6</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>2</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>5</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0.2857142857142857</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>3</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>6</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>1</d4p1:Value>\r\n          </d4p1:Node>\r\n        </d4p1:Nodes>\r\n        <d4p1:Probabilities xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"16\" z:Size=\"4\">\r\n          <d5p1:ArrayOfdouble z:Id=\"17\" z:Size=\"0\" />\r\n          <d5p1:ArrayOfdouble z:Id=\"18\" z:Size=\"0\" />\r\n          <d5p1:ArrayOfdouble z:Id=\"19\" z:Size=\"0\" />\r\n          <d5p1:ArrayOfdouble z:Id=\"20\" z:Size=\"0\" />\r\n        </d4p1:Probabilities>\r\n        <d4p1:TargetNames xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"21\" z:Size=\"2\">\r\n          <d5p1:double>0</d5p1:double>\r\n          <d5p1:double>1</d5p1:double>\r\n        </d4p1:TargetNames>\r\n        <d4p1:VariableImportance xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"22\" z:Size=\"2\">\r\n          <d5p1:double>0.13296703296703297</d5p1:double>\r\n          <d5p1:double>2.448260073260073</d5p1:double>\r\n        </d4p1:VariableImportance>\r\n      </d2p1:Tree>\r\n      <d2p1:m_variableImportance xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"22\" i:nil=\"true\" />\r\n    </d2p1:RegressionDecisionTreeModel>\r\n  </_x003C_Trees_x003E_k__BackingField>\r\n  <m_rawVariableImportance xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"23\" z:Size=\"2\">\r\n    <d2p1:double>0.89030969030969</d2p1:double>\r\n    <d2p1:double>2.85540293040293</d2p1:double>\r\n  </m_rawVariableImportance>\r\n</RegressionForestModel>";

    [TestMethod]
    public void RegressionForestModel_Predict_Single()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new RegressionRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, false);
        var sut = learner.Learn(observations, targets);

        var rows = targets.Length;
        var predictions = new double[rows];
        for (var i = 0; i < rows; i++)
        {
            predictions[i] = sut.Predict(observations.Row(i));
        }

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var error = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.15381141277554411, error, m_delta);
    }

    [TestMethod]
    public void RegressionForestModel_Predict_Multiple()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new RegressionRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, false);
        var sut = learner.Learn(observations, targets);

        var predictions = sut.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var error = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.15381141277554411, error, m_delta);
    }

    [TestMethod]
    public void RegressionForestModel_PredictCertainty_Single()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new RegressionRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, false);
        var sut = learner.Learn(observations, targets);

        var rows = targets.Length;
        var actual = new CertaintyPrediction[rows];
        for (var i = 0; i < rows; i++)
        {
            actual[i] = sut.PredictCertainty(observations.Row(i));
        }

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

        Assert.AreEqual(0.15381141277554411, error, m_delta);

        var expected = new CertaintyPrediction[] { new(0.379151515151515, 0.0608255007215007), new(0.411071351850763, 0.0831655436577049), new(0.243420918950331, 0.0452827034233046), new(0.302332251082251, 0.0699917594408057), new(0.411071351850763, 0.0831655436577049), new(0.175743762773174, 0.0354069437824887), new(0.574083361083361, 0.0765858693929188), new(0.259063776093188, 0.0491198812971218), new(0.163878898878899, 0.0331543420321184), new(0.671753996003996, 0.0624466591504497), new(0.418472943722944, 0.0607014359023913), new(0.243420918950331, 0.0452827034233046), new(0.443779942279942, 0.0941961872991865), new(0.156999361749362, 0.0435804333960299), new(0.591222034501446, 0.0873624628347336), new(0.123822406351818, 0.0283119805431255), new(0.162873993653405, 0.0333697457759022), new(0.596261932511932, 0.0695341060210394), new(0.671753996003996, 0.0624466591504497), new(0.418472943722944, 0.0607014359023913), new(0.329000027750028, 0.0788869852405852), new(0.671753996003996, 0.0624466591504497), new(0.499770375049787, 0.0913884936411888), new(0.140025508804921, 0.0309875116490099), new(0.161207326986739, 0.0336321035325246), new(0.389553418803419, 0.0744433596104835), };
        CollectionAssert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void RegressionForestModel_PredictProbability_Multiple()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new RegressionRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, false);
        var sut = learner.Learn(observations, targets);
        var actual = sut.PredictCertainty(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

        Assert.AreEqual(0.15381141277554411, error, m_delta);

        var expected = new CertaintyPrediction[] { new(0.379151515151515, 0.0608255007215007), new(0.411071351850763, 0.0831655436577049), new(0.243420918950331, 0.0452827034233046), new(0.302332251082251, 0.0699917594408057), new(0.411071351850763, 0.0831655436577049), new(0.175743762773174, 0.0354069437824887), new(0.574083361083361, 0.0765858693929188), new(0.259063776093188, 0.0491198812971218), new(0.163878898878899, 0.0331543420321184), new(0.671753996003996, 0.0624466591504497), new(0.418472943722944, 0.0607014359023913), new(0.243420918950331, 0.0452827034233046), new(0.443779942279942, 0.0941961872991865), new(0.156999361749362, 0.0435804333960299), new(0.591222034501446, 0.0873624628347336), new(0.123822406351818, 0.0283119805431255), new(0.162873993653405, 0.0333697457759022), new(0.596261932511932, 0.0695341060210394), new(0.671753996003996, 0.0624466591504497), new(0.418472943722944, 0.0607014359023913), new(0.329000027750028, 0.0788869852405852), new(0.671753996003996, 0.0624466591504497), new(0.499770375049787, 0.0913884936411888), new(0.140025508804921, 0.0309875116490099), new(0.161207326986739, 0.0336321035325246), new(0.389553418803419, 0.0744433596104835), };
        CollectionAssert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void RegressionForestModel_Trees()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var reader = new StringReader(m_regressionForestModelString);
        var sut = RegressionForestModel.Load(() => reader);

        var rows = observations.RowCount;
        var predictions = new double[rows];
        for (var row = 0; row < rows; row++)
        {
            var observation = observations.Row(row);
            predictions[row] = sut.Trees.Select(t => t.Predict(observation))
                .Average();
        }

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var error = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.14547628738104926, error, m_delta);
    }

    [TestMethod]
    public void RegressionForestModel_GetVariableImportance()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 },
            { "PreviousExperience_month", 1 } };

        var learner = new RegressionRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, false);
        var sut = learner.Learn(observations, targets);

        var actual = sut.GetVariableImportance(featureNameToIndex);
        var expected = new Dictionary<string, double> { {"PreviousExperience_month", 100},
            {"AptitudeTestScore", 42.3879919692465 }};

        Assert.AreEqual(expected.Count, actual.Count);
        var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

        foreach (var item in zip)
        {
            Assert.AreEqual(item.Expected.Key, item.Actual.Key);
            Assert.AreEqual(item.Expected.Value, item.Actual.Value, m_delta);
        }
    }

    [TestMethod]
    public void RegressionForestModel_GetRawVariableImportance()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new RegressionRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, false);
        var sut = learner.Learn(observations, targets);

        var actual = sut.GetRawVariableImportance();
        var expected = new double[] { 59.2053755086635, 139.67487667643803 };

        Assert.AreEqual(expected.Length, actual.Length);

        for (var i = 0; i < expected.Length; i++)
        {
            Assert.AreEqual(expected[i], actual[i], m_delta);
        }
    }

    [TestMethod]
    public void RegressionForestModel_Save()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learner = new RegressionRandomForestLearner(2, 5, 100, 1, 0.0001, 1.0, 42, false);
        var sut = learner.Learn(observations, targets);

        // save model.
        var writer = new StringWriter();
        sut.Save(() => writer);

        // load model and assert prediction results.
        sut = RegressionForestModel.Load(() => new StringReader(writer.ToString()));
        var predictions = sut.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.14547628738104926, actual, m_delta);
    }

    [TestMethod]
    public void RegressionForestModel_Load()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var reader = new StringReader(m_regressionForestModelString);
        var sut = RegressionForestModel.Load(() => reader);

        var predictions = sut.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var error = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.14547628738104926, error, m_delta);
    }

    static void Write(CertaintyPrediction[] predictions)
    {
        var value = "new CertaintyPrediction[] {";
        foreach (var item in predictions)
        {
            value += "new CertaintyPrediction(" + item.Prediction + ", " + item.Variance + "), ";
        }

        value += "};";

        Trace.WriteLine(value);
    }
}
