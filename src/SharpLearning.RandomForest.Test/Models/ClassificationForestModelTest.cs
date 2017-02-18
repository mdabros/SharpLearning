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

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, false);
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

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, false);
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

            var learner = new ClassificationRandomForestLearner(100, 1, 100, 1, 0.0001, 1.0, 42, false);
            var sut = learner.Learn(observations, targets);

            var actual = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = sut.PredictProbability(observations.Row(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.076923076923076927, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.650149027443145 }, { 1, 0.349850972556855 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.566943847818848 }, { 1, 0.433056152181152 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.726936489980608 }, { 1, 0.273063510019392 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.752781908451026 }, { 1, 0.247218091548974 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.566943847818848 }, { 1, 0.433056152181152 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.792506836300954 }, { 1, 0.207493163699046 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.491736055611056 }, { 1, 0.508263944388944 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.574583315377433 }, { 1, 0.425416684622567 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.838724674018791 }, { 1, 0.161275325981208 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.241480824730825 }, { 1, 0.758519175269175 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.385258186258186 }, { 1, 0.614741813741813 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.726936489980608 }, { 1, 0.273063510019392 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.706733044733045 }, { 1, 0.293266955266955 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.801266011766012 }, { 1, 0.198733988233988 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.294952297702298 }, { 1, 0.705047702297702 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.821706914001031 }, { 1, 0.178293085998968 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.780062391856509 }, { 1, 0.21993760814349 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.554444388944389 }, { 1, 0.445555611055611 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.261349872349872 }, { 1, 0.738650127650127 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.419758186258186 }, { 1, 0.580241813741813 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.71382231249143 }, { 1, 0.28617768750857 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.241480824730825 }, { 1, 0.758519175269175 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.47562148962149 }, { 1, 0.52437851037851 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.821706914001031 }, { 1, 0.178293085998968 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.792506836300954 }, { 1, 0.207493163699046 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.666244987039105 }, { 1, 0.333755012960895 }, }) };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationForestModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 1, 100, 1, 0.0001, 1.0, 42, false);
            var sut = learner.Learn(observations, targets);

            var actual = sut.PredictProbability(observations);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.076923076923076927, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.650149027443145 }, { 1, 0.349850972556855 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.566943847818848 }, { 1, 0.433056152181152 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.726936489980608 }, { 1, 0.273063510019392 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.752781908451026 }, { 1, 0.247218091548974 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.566943847818848 }, { 1, 0.433056152181152 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.792506836300954 }, { 1, 0.207493163699046 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.491736055611056 }, { 1, 0.508263944388944 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.574583315377433 }, { 1, 0.425416684622567 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.838724674018791 }, { 1, 0.161275325981208 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.241480824730825 }, { 1, 0.758519175269175 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.385258186258186 }, { 1, 0.614741813741813 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.726936489980608 }, { 1, 0.273063510019392 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.706733044733045 }, { 1, 0.293266955266955 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.801266011766012 }, { 1, 0.198733988233988 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.294952297702298 }, { 1, 0.705047702297702 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.821706914001031 }, { 1, 0.178293085998968 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.780062391856509 }, { 1, 0.21993760814349 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.554444388944389 }, { 1, 0.445555611055611 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.261349872349872 }, { 1, 0.738650127650127 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.419758186258186 }, { 1, 0.580241813741813 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.71382231249143 }, { 1, 0.28617768750857 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.241480824730825 }, { 1, 0.758519175269175 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.47562148962149 }, { 1, 0.52437851037851 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.821706914001031 }, { 1, 0.178293085998968 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.792506836300954 }, { 1, 0.207493163699046 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.666244987039105 }, { 1, 0.333755012960895 }, }) };
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

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, false);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { {"PreviousExperience_month", 100},
                {"AptitudeTestScore", 43.4356891141648 }};

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

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, false);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 5.1708306492004992, 11.904566854251304 };

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

            var learner = new ClassificationRandomForestLearner(2, 5, 100, 1, 0.0001, 1.0, 42, false);
            var sut = learner.Learn(observations, targets);

            var writer = new StringWriter();
            sut.Save(() => writer);

            var actual = writer.ToString();
            Assert.AreEqual(ClassificationForestModelString, actual);
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

            Assert.AreEqual(0.23076923076923078, error, 0.0000001);
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
            "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<ClassificationForestModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.RandomForest.Models\">\r\n  <m_models xmlns:d2p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Models\" z:Id=\"2\" z:Size=\"2\">\r\n    <d2p1:ClassificationDecisionTreeModel z:Id=\"3\">\r\n      <d2p1:Tree xmlns:d4p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\" z:Id=\"4\">\r\n        <d4p1:Nodes z:Id=\"5\" z:Size=\"7\">\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>0</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>6</d4p1:RightIndex>\r\n            <d4p1:Value>20</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>2</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>1</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>3</d4p1:RightIndex>\r\n            <d4p1:Value>9.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>0</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>2</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>0</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>4</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>3</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>5</d4p1:RightIndex>\r\n            <d4p1:Value>2</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>4</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>1</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>2</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>5</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>3</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>6</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>1</d4p1:Value>\r\n          </d4p1:Node>\r\n        </d4p1:Nodes>\r\n        <d4p1:Probabilities xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"6\" z:Size=\"4\">\r\n          <d5p1:ArrayOfdouble z:Id=\"7\" z:Size=\"2\">\r\n            <d5p1:double>0.81818181818181823</d5p1:double>\r\n            <d5p1:double>0.18181818181818182</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n          <d5p1:ArrayOfdouble z:Id=\"8\" z:Size=\"2\">\r\n            <d5p1:double>0.42857142857142855</d5p1:double>\r\n            <d5p1:double>0.5714285714285714</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n          <d5p1:ArrayOfdouble z:Id=\"9\" z:Size=\"2\">\r\n            <d5p1:double>0.66666666666666663</d5p1:double>\r\n            <d5p1:double>0.33333333333333331</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n          <d5p1:ArrayOfdouble z:Id=\"10\" z:Size=\"2\">\r\n            <d5p1:double>0.14285714285714285</d5p1:double>\r\n            <d5p1:double>0.8571428571428571</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n        </d4p1:Probabilities>\r\n        <d4p1:TargetNames xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"11\" z:Size=\"2\">\r\n          <d5p1:double>0</d5p1:double>\r\n          <d5p1:double>1</d5p1:double>\r\n        </d4p1:TargetNames>\r\n        <d4p1:VariableImportance xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"12\" z:Size=\"2\">\r\n          <d5p1:double>0.022161172161172173</d5p1:double>\r\n          <d5p1:double>0.19543063773833011</d5p1:double>\r\n        </d4p1:VariableImportance>\r\n      </d2p1:Tree>\r\n      <d2p1:m_variableImportance xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"12\" i:nil=\"true\" />\r\n    </d2p1:ClassificationDecisionTreeModel>\r\n    <d2p1:ClassificationDecisionTreeModel z:Id=\"13\">\r\n      <d2p1:Tree xmlns:d4p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\" z:Id=\"14\">\r\n        <d4p1:Nodes z:Id=\"15\" z:Size=\"7\">\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>0</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>0</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>4</d4p1:RightIndex>\r\n            <d4p1:Value>3.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>2</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>1</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>3</d4p1:RightIndex>\r\n            <d4p1:Value>13.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>0</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>2</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>3</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>5</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>4</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>6</d4p1:RightIndex>\r\n            <d4p1:Value>17</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>2</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>5</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>1</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>3</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>6</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0</d4p1:Value>\r\n          </d4p1:Node>\r\n        </d4p1:Nodes>\r\n        <d4p1:Probabilities xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"16\" z:Size=\"4\">\r\n          <d5p1:ArrayOfdouble z:Id=\"17\" z:Size=\"2\">\r\n            <d5p1:double>0.9</d5p1:double>\r\n            <d5p1:double>0.1</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n          <d5p1:ArrayOfdouble z:Id=\"18\" z:Size=\"2\">\r\n            <d5p1:double>0.55555555555555558</d5p1:double>\r\n            <d5p1:double>0.44444444444444442</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n          <d5p1:ArrayOfdouble z:Id=\"19\" z:Size=\"2\">\r\n            <d5p1:double>0.42857142857142855</d5p1:double>\r\n            <d5p1:double>0.5714285714285714</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n          <d5p1:ArrayOfdouble z:Id=\"20\" z:Size=\"2\">\r\n            <d5p1:double>0.5</d5p1:double>\r\n            <d5p1:double>0.5</d5p1:double>\r\n          </d5p1:ArrayOfdouble>\r\n        </d4p1:Probabilities>\r\n        <d4p1:TargetNames xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"21\" z:Size=\"2\">\r\n          <d5p1:double>0</d5p1:double>\r\n          <d5p1:double>1</d5p1:double>\r\n        </d4p1:TargetNames>\r\n        <d4p1:VariableImportance xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"22\" z:Size=\"2\">\r\n          <d5p1:double>0.058257127487896736</d5p1:double>\r\n          <d5p1:double>0.054845154845154849</d5p1:double>\r\n        </d4p1:VariableImportance>\r\n      </d2p1:Tree>\r\n      <d2p1:m_variableImportance xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"22\" i:nil=\"true\" />\r\n    </d2p1:ClassificationDecisionTreeModel>\r\n  </m_models>\r\n  <m_rawVariableImportance xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"23\" z:Size=\"2\">\r\n    <d2p1:double>0.080418299649068908</d2p1:double>\r\n    <d2p1:double>0.250275792583485</d2p1:double>\r\n  </m_rawVariableImportance>\r\n</ClassificationForestModel>";
    }
}
