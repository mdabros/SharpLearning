using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.RandomForest.Test.Properties;
using SharpLearning.RandomForest.Learners;
using SharpLearning.Metrics.Classification;
using SharpLearning.Containers;
using System.Linq;
using System.Diagnostics;
using SharpLearning.RandomForest.Models;

namespace SharpLearning.RandomForest.Test.Models
{
    [TestClass]
    public class ClassificationForestModelTest
    {
        [TestMethod]
        public void ClassificationForestModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.Row(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.23076923076923078, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationForestModel_Predict_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.23076923076923078, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationForestModel_PredictProbability_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 1, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = sut.PredictProbability(observations.Row(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.076923076923076927, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.659019957381799 }, { 1, 0.340980042618201 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.59688515242927 }, { 1, 0.40311484757073 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.719480409511369 }, { 1, 0.280519590488631 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.744205045986005 }, { 1, 0.255794954013994 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.59688515242927 }, { 1, 0.40311484757073 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.788828061859021 }, { 1, 0.211171938140978 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.495583888333888 }, { 1, 0.504416111666112 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.58936533014629 }, { 1, 0.41063466985371 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.830597486878446 }, { 1, 0.169402513121553 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.256507450882451 }, { 1, 0.743492549117549 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.349079795204795 }, { 1, 0.650920204795204 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.719480409511369 }, { 1, 0.280519590488631 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.681955044955045 }, { 1, 0.318044955044955 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.774851926882887 }, { 1, 0.225148073117113 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.304978632478632 }, { 1, 0.695021367521367 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.821320069851029 }, { 1, 0.17867993014897 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.788828061859021 }, { 1, 0.211171938140978 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.604202935952936 }, { 1, 0.395797064047064 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.277229673104673 }, { 1, 0.722770326895327 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.392208491070333 }, { 1, 0.607791508929667 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.717849096380056 }, { 1, 0.282150903619944 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.256507450882451 }, { 1, 0.743492549117549 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.452195415695416 }, { 1, 0.547804584304584 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.821320069851029 }, { 1, 0.17867993014897 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.788828061859021 }, { 1, 0.211171938140978 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.654524036635878 }, { 1, 0.345475963364121 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationForestModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 1, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.PredictProbability(observations);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.076923076923076927, error, 0.0000001);
            
            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.659019957381799 }, { 1, 0.340980042618201 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.59688515242927 }, { 1, 0.40311484757073 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.719480409511369 }, { 1, 0.280519590488631 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.744205045986005 }, { 1, 0.255794954013994 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.59688515242927 }, { 1, 0.40311484757073 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.788828061859021 }, { 1, 0.211171938140978 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.495583888333888 }, { 1, 0.504416111666112 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.58936533014629 }, { 1, 0.41063466985371 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.830597486878446 }, { 1, 0.169402513121553 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.256507450882451 }, { 1, 0.743492549117549 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.349079795204795 }, { 1, 0.650920204795204 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.719480409511369 }, { 1, 0.280519590488631 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.681955044955045 }, { 1, 0.318044955044955 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.774851926882887 }, { 1, 0.225148073117113 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.304978632478632 }, { 1, 0.695021367521367 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.821320069851029 }, { 1, 0.17867993014897 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.788828061859021 }, { 1, 0.211171938140978 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.604202935952936 }, { 1, 0.395797064047064 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.277229673104673 }, { 1, 0.722770326895327 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.392208491070333 }, { 1, 0.607791508929667 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.717849096380056 }, { 1, 0.282150903619944 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.256507450882451 }, { 1, 0.743492549117549 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.452195415695416 }, { 1, 0.547804584304584 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.821320069851029 }, { 1, 0.17867993014897 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.788828061859021 }, { 1, 0.211171938140978 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.654524036635878 }, { 1, 0.345475963364121 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationForestModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, 
                { "AptitudeTestScore", 42.4462981352022 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void ClassificationForestModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 4.5932480752556968, 10.821316055937396 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        [TestMethod]
        public void ClassificationForestModel_Save()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new ClassificationRandomForestLearner(2, 5, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var writer = new StringWriter();
            sut.Save(() => writer);

            Assert.AreEqual(ClassificationForestModelString, writer.ToString());
        }

        [TestMethod]
        public void ClassificationForestModel_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var reader = new StringReader(ClassificationForestModelString);
            var sut = ClassificationForestModel.Load(() => reader);

            var predictions = sut.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.38461538461538464, error, 0.0000001);
        }

        void Write(ProbabilityPrediction[] predictions)
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

        readonly string ClassificationForestModelString =
            "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<ClassificationForestModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.RandomForest.Models\">\r\n  <m_models xmlns:d2p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Models\" z:Id=\"2\" z:Size=\"2\">\r\n    <d2p1:ClassificationDecisionTreeModel z:Id=\"3\">\r\n      <d2p1:Tree xmlns:d4p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\" z:Id=\"4\">\r\n        <d4p1:Nodes z:Id=\"5\" z:Size=\"7\">\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>0</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>4</d4p1:RightIndex>\r\n            <d4p1:Value>9.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>2</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>1</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>3</d4p1:RightIndex>\r\n            <d4p1:Value>4.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>0</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>2</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>1</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>3</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>0</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>5</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>4</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>6</d4p1:RightIndex>\r\n            <d4p1:Value>3</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>2</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>5</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>1</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>3</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>6</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0</d4p1:Value>\r\n          </d4p1:Node>\r\n        </d4p1:Nodes>\r\n        <d4p1:Probabilities xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"6\" z:Size=\"4\">\r\n          <d5p1:ArrayOfdouble z:Id=\"7\" z:Size=\"2\">\r\n            <d5p1:double>0.375</d5p1:double>\r\n            <d5p1:double>0.625</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n          <d5p1:ArrayOfdouble z:Id=\"8\" z:Size=\"2\">\r\n            <d5p1:double>0.88888888888888884</d5p1:double>\r\n            <d5p1:double>0.11111111111111111</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n          <d5p1:ArrayOfdouble z:Id=\"9\" z:Size=\"2\">\r\n            <d5p1:double>0.11111111111111111</d5p1:double>\r\n            <d5p1:double>0.88888888888888884</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n          <d5p1:ArrayOfdouble z:Id=\"10\" z:Size=\"2\">\r\n            <d5p1:double>0.75</d5p1:double>\r\n            <d5p1:double>0.25</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n        </d4p1:Probabilities>\r\n        <d4p1:TargetNames xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"11\" z:Size=\"2\">\r\n          <d5p1:double>0</d5p1:double>\r\n          <d5p1:double>1</d5p1:double>\r\n        </d4p1:TargetNames>\r\n        <d4p1:VariableImportance xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"12\" z:Size=\"2\">\r\n          <d5p1:double>0.17258382642998027</d5p1:double>\r\n          <d5p1:double>0.15779092702169625</d5p1:double>\r\n        </d4p1:VariableImportance>\r\n      </d2p1:Tree>\r\n      <d2p1:m_variableImportance xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"12\" i:nil=\"true\" />\r\n    </d2p1:ClassificationDecisionTreeModel>\r\n    <d2p1:ClassificationDecisionTreeModel z:Id=\"13\">\r\n      <d2p1:Tree xmlns:d4p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\" z:Id=\"14\">\r\n        <d4p1:Nodes z:Id=\"15\" z:Size=\"7\">\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>0</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>0</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>4</d4p1:RightIndex>\r\n            <d4p1:Value>3.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>2</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>1</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>3</d4p1:RightIndex>\r\n            <d4p1:Value>13.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>0</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>2</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>3</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>5</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>4</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>6</d4p1:RightIndex>\r\n            <d4p1:Value>17</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>2</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>5</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>1</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>3</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>6</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0</d4p1:Value>\r\n          </d4p1:Node>\r\n        </d4p1:Nodes>\r\n        <d4p1:Probabilities xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"16\" z:Size=\"4\">\r\n          <d5p1:ArrayOfdouble z:Id=\"17\" z:Size=\"2\">\r\n            <d5p1:double>0.9</d5p1:double>\r\n            <d5p1:double>0.1</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n          <d5p1:ArrayOfdouble z:Id=\"18\" z:Size=\"2\">\r\n            <d5p1:double>0.55555555555555558</d5p1:double>\r\n            <d5p1:double>0.44444444444444442</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n          <d5p1:ArrayOfdouble z:Id=\"19\" z:Size=\"2\">\r\n            <d5p1:double>0.42857142857142855</d5p1:double>\r\n            <d5p1:double>0.5714285714285714</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n          <d5p1:ArrayOfdouble z:Id=\"20\" z:Size=\"2\">\r\n            <d5p1:double>0.5</d5p1:double>\r\n            <d5p1:double>0.5</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n        </d4p1:Probabilities>\r\n        <d4p1:TargetNames xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"21\" z:Size=\"2\">\r\n          <d5p1:double>0</d5p1:double>\r\n          <d5p1:double>1</d5p1:double>\r\n        </d4p1:TargetNames>\r\n        <d4p1:VariableImportance xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"22\" z:Size=\"2\">\r\n          <d5p1:double>0.058257127487896736</d5p1:double>\r\n          <d5p1:double>0.054845154845154849</d5p1:double>\r\n        </d4p1:VariableImportance>\r\n      </d2p1:Tree>\r\n      <d2p1:m_variableImportance xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"22\" i:nil=\"true\" />\r\n    </d2p1:ClassificationDecisionTreeModel>\r\n  </m_models>\r\n  <m_rawVariableImportance xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"23\" z:Size=\"2\">\r\n    <d2p1:double>0.230840953917877</d2p1:double>\r\n    <d2p1:double>0.2126360818668511</d2p1:double>\r\n  </m_rawVariableImportance>\r\n</ClassificationForestModel>";
    }
}
