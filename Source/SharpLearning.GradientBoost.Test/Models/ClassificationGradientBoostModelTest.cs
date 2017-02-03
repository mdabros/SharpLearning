using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using System.Linq;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.Metrics.Classification;
using SharpLearning.Containers;
using System.Collections.Generic;
using System.Diagnostics;
using SharpLearning.GradientBoost.Models;
using SharpLearning.GradientBoost.Learners;

namespace SharpLearning.GradientBoost.Test.Models
{
    [TestClass]
    public class ClassificationGradientBoostModelTest
    {
        [TestMethod]
        public void ClassificationGradientBoostModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationGradientBoostLearner();
            var sut = learner.Learn(observations, targets);

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
        public void ClassificationGradientBoostModel_Predict_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationGradientBoostLearner();
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationGradientBoostModel_PredictProbability_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationGradientBoostLearner();
            var sut = learner.Learn(observations, targets);

            var actual = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = sut.PredictProbability(observations.Row(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00153419685769873 }, { 0, 0.998465803142301 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.497135615200052 }, { 0, 0.502864384799948 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00674291737944022 }, { 0, 0.99325708262056 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00153419685769873 }, { 0, 0.998465803142301 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.497135615200052 }, { 0, 0.502864384799948 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00428497228545111 }, { 0, 0.995715027714549 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.987907185249206 }, { 0, 0.0120928147507945 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.982783250692275 }, { 0, 0.0172167493077254 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00262490179961228 }, { 0, 0.997375098200388 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.996417847055106 }, { 0, 0.00358215294489364 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.995341658753364 }, { 0, 0.00465834124663571 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00674291737944022 }, { 0, 0.99325708262056 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.0118633115475969 }, { 0, 0.988136688452403 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00048646805791186 }, { 0, 0.999513531942088 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.999891769651047 }, { 0, 0.000108230348952856 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00334655581934884 }, { 0, 0.996653444180651 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00428497228545111 }, { 0, 0.995715027714549 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.0118633115475969 }, { 0, 0.988136688452403 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.996417847055106 }, { 0, 0.00358215294489362 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.993419876193791 }, { 0, 0.00658012380620933 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00262490179961228 }, { 0, 0.997375098200388 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.996417847055106 }, { 0, 0.00358215294489362 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.988568859753437 }, { 0, 0.0114311402465632 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00334655581934884 }, { 0, 0.996653444180651 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00428497228545111 }, { 0, 0.995715027714549 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00262490179961228 }, { 0, 0.997375098200388 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationGradientBoostModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationGradientBoostLearner();
            var sut = learner.Learn(observations, targets);

            var actual = sut.PredictProbability(observations);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00153419685769873 }, { 0, 0.998465803142301 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.497135615200052 }, { 0, 0.502864384799948 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00674291737944022 }, { 0, 0.99325708262056 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00153419685769873 }, { 0, 0.998465803142301 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.497135615200052 }, { 0, 0.502864384799948 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00428497228545111 }, { 0, 0.995715027714549 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.987907185249206 }, { 0, 0.0120928147507945 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.982783250692275 }, { 0, 0.0172167493077254 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00262490179961228 }, { 0, 0.997375098200388 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.996417847055106 }, { 0, 0.00358215294489364 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.995341658753364 }, { 0, 0.00465834124663571 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00674291737944022 }, { 0, 0.99325708262056 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.0118633115475969 }, { 0, 0.988136688452403 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00048646805791186 }, { 0, 0.999513531942088 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.999891769651047 }, { 0, 0.000108230348952856 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00334655581934884 }, { 0, 0.996653444180651 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00428497228545111 }, { 0, 0.995715027714549 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.0118633115475969 }, { 0, 0.988136688452403 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.996417847055106 }, { 0, 0.00358215294489362 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.993419876193791 }, { 0, 0.00658012380620933 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00262490179961228 }, { 0, 0.997375098200388 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.996417847055106 }, { 0, 0.00358215294489362 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.988568859753437 }, { 0, 0.0114311402465632 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00334655581934884 }, { 0, 0.996653444180651 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00428497228545111 }, { 0, 0.995715027714549 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.00262490179961228 }, { 0, 0.997375098200388 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationGradientBoostModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learner = new ClassificationGradientBoostLearner();
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, 
                { "AptitudeTestScore", 56.81853305612 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void ClassificationGradientBoostModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new ClassificationGradientBoostLearner();
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 26.287331114005394, 46.265416757664667 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.001);
            }
        }

        [TestMethod]
        public void ClassificationGradientBoostModel_Save()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new ClassificationGradientBoostLearner(5);
            var sut = learner.Learn(observations, targets);

            var writer = new StringWriter();
            sut.Save(() => writer);

            var actual = writer.ToString();
            Assert.AreEqual(ClassificationGradientBoostModelString, actual);
        }

        [TestMethod]
        public void ClassificationGradientBoostModel_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var reader = new StringReader(ClassificationGradientBoostModelString);
            var sut = ClassificationGradientBoostModel.Load(() => reader);

            var predictions = sut.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.15384615384615385, error, 0.0000001);
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

        readonly string ClassificationGradientBoostModelString =
        "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<ClassificationGradientBoostModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.GradientBoost.Models\">\r\n  <FeatureCount>2</FeatureCount>\r\n  <InitialLoss>-0.47000362924573558</InitialLoss>\r\n  <LearningRate>0.1</LearningRate>\r\n  <TargetNames xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"2\" z:Size=\"2\">\r\n    <d2p1:double>0</d2p1:double>\r\n    <d2p1:double>1</d2p1:double>\r\n  </TargetNames>\r\n  <Trees xmlns:d2p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.GradientBoost.GBMDecisionTree\" z:Id=\"3\" z:Size=\"1\">\r\n    <d2p1:ArrayOfGBMTree z:Id=\"4\" z:Size=\"5\">\r\n      <d2p1:GBMTree z:Id=\"5\">\r\n        <d2p1:Nodes z:Id=\"6\" z:Size=\"5\">\r\n          <d2p1:GBMNode z:Id=\"7\">\r\n            <d2p1:Depth>0</d2p1:Depth>\r\n            <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>0.9749999999999992</d2p1:LeftConstant>\r\n            <d2p1:LeftError>6.1538461538461577</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>0.9749999999999992</d2p1:RightConstant>\r\n            <d2p1:RightError>6.1538461538461577</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>26</d2p1:SampleCount>\r\n            <d2p1:SplitValue>-1</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"8\">\r\n            <d2p1:Depth>1</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>1.4477272727272721</d2p1:LeftConstant>\r\n            <d2p1:LeftError>4.3636363636363686</d2p1:LeftError>\r\n            <d2p1:LeftIndex>2</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.625000000000002</d2p1:RightConstant>\r\n            <d2p1:RightError>-1.5543122344752192E-15</d2p1:RightError>\r\n            <d2p1:RightIndex>3</d2p1:RightIndex>\r\n            <d2p1:SampleCount>26</d2p1:SampleCount>\r\n            <d2p1:SplitValue>20</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"9\">\r\n            <d2p1:Depth>2</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>-1.625</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>1.5940476190476185</d2p1:RightConstant>\r\n            <d2p1:RightError>3.8095238095238142</d2p1:RightError>\r\n            <d2p1:RightIndex>4</d2p1:RightIndex>\r\n            <d2p1:SampleCount>22</d2p1:SampleCount>\r\n            <d2p1:SplitValue>2.5</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"10\">\r\n            <d2p1:Depth>2</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>-1.625</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.6250000000000027</d2p1:RightConstant>\r\n            <d2p1:RightError>-1.609823385706477E-15</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>4</d2p1:SampleCount>\r\n            <d2p1:SplitValue>23</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"11\">\r\n            <d2p1:Depth>3</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>2.0366666666666666</d2p1:LeftConstant>\r\n            <d2p1:LeftError>1.7333333333333374</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>0.4874999999999996</d2p1:RightConstant>\r\n            <d2p1:RightError>1.5</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>21</d2p1:SampleCount>\r\n            <d2p1:SplitValue>13.5</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n        </d2p1:Nodes>\r\n      </d2p1:GBMTree>\r\n      <d2p1:GBMTree z:Id=\"12\">\r\n        <d2p1:Nodes z:Id=\"13\" z:Size=\"6\">\r\n          <d2p1:GBMNode z:Id=\"14\">\r\n            <d2p1:Depth>0</d2p1:Depth>\r\n            <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>0.86059480408450362</d2p1:LeftConstant>\r\n            <d2p1:LeftError>5.596711778030242</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>0.86059480408450362</d2p1:RightConstant>\r\n            <d2p1:RightError>5.596711778030242</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>26</d2p1:SampleCount>\r\n            <d2p1:SplitValue>-1</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"15\">\r\n            <d2p1:Depth>1</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>1.266063090014917</d2p1:LeftConstant>\r\n            <d2p1:LeftError>4.1463589231846409</d2p1:LeftError>\r\n            <d2p1:LeftIndex>2</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.5312600563908738</d2p1:RightConstant>\r\n            <d2p1:RightError>-1.27675647831893E-15</d2p1:RightError>\r\n            <d2p1:RightIndex>3</d2p1:RightIndex>\r\n            <d2p1:SampleCount>26</d2p1:SampleCount>\r\n            <d2p1:SplitValue>20</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"16\">\r\n            <d2p1:Depth>2</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>-0.2989037549136308</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0.6137878335116782</d2p1:LeftError>\r\n            <d2p1:LeftIndex>4</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>1.5087703239250865</d2p1:RightConstant>\r\n            <d2p1:RightError>3.0331204175116282</d2p1:RightError>\r\n            <d2p1:RightIndex>5</d2p1:RightIndex>\r\n            <d2p1:SampleCount>22</d2p1:SampleCount>\r\n            <d2p1:SplitValue>4.5</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"17\">\r\n            <d2p1:Depth>2</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>-1.5312600563908738</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.5312600563908738</d2p1:RightConstant>\r\n            <d2p1:RightError>-1.27675647831893E-15</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>4</d2p1:SampleCount>\r\n            <d2p1:SplitValue>23</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"18\">\r\n            <d2p1:Depth>3</d2p1:Depth>\r\n            <d2p1:FeatureIndex>0</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>2.3051747796578996</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.6534579988148759</d2p1:RightConstant>\r\n            <d2p1:RightError>0.0037726356416253326</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>3</d2p1:SampleCount>\r\n            <d2p1:SplitValue>4</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"19\">\r\n            <d2p1:Depth>3</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>1.9919935727839668</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0.92307692307692335</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>0.43382354239469473</d2p1:RightConstant>\r\n            <d2p1:RightError>1.4999999999999982</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>19</d2p1:SampleCount>\r\n            <d2p1:SplitValue>13.5</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n        </d2p1:Nodes>\r\n      </d2p1:GBMTree>\r\n      <d2p1:GBMTree z:Id=\"20\">\r\n        <d2p1:Nodes z:Id=\"21\" z:Size=\"6\">\r\n          <d2p1:GBMNode z:Id=\"22\">\r\n            <d2p1:Depth>0</d2p1:Depth>\r\n            <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>0.772024071047949</d2p1:LeftConstant>\r\n            <d2p1:LeftError>4.9956325710355829</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>0.772024071047949</d2p1:RightConstant>\r\n            <d2p1:RightError>4.9956325710355829</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>26</d2p1:SampleCount>\r\n            <d2p1:SplitValue>-1</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"23\">\r\n            <d2p1:Depth>1</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>1.1271620452174398</d2p1:LeftConstant>\r\n            <d2p1:LeftError>3.81901068866107</d2p1:LeftError>\r\n            <d2p1:LeftIndex>2</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.4558326033028339</d2p1:RightConstant>\r\n            <d2p1:RightError>-5.5511151231257827E-17</d2p1:RightError>\r\n            <d2p1:RightIndex>3</d2p1:RightIndex>\r\n            <d2p1:SampleCount>26</d2p1:SampleCount>\r\n            <d2p1:SplitValue>20</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"24\">\r\n            <d2p1:Depth>2</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>-0.27786210243575493</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0.49774616548974571</d2p1:LeftError>\r\n            <d2p1:LeftIndex>4</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>1.3374975310021555</d2p1:RightConstant>\r\n            <d2p1:RightError>2.9159510952504579</d2p1:RightError>\r\n            <d2p1:RightIndex>5</d2p1:RightIndex>\r\n            <d2p1:SampleCount>22</d2p1:SampleCount>\r\n            <d2p1:SplitValue>4.5</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"25\">\r\n            <d2p1:Depth>2</d2p1:Depth>\r\n            <d2p1:FeatureIndex>0</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>-1.4558326033028344</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.4558326033028339</d2p1:RightConstant>\r\n            <d2p1:RightError>-1.1102230246251565E-16</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>4</d2p1:SampleCount>\r\n            <d2p1:SplitValue>2.5</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"26\">\r\n            <d2p1:Depth>3</d2p1:Depth>\r\n            <d2p1:FeatureIndex>0</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>2.0364687310490139</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.5552695523522238</d2p1:RightConstant>\r\n            <d2p1:RightError>0.0034643522773981916</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>3</d2p1:SampleCount>\r\n            <d2p1:SplitValue>4</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"27\">\r\n            <d2p1:Depth>3</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>1.7614034635323572</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0.923076923076926</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>0.38692819292266051</d2p1:RightConstant>\r\n            <d2p1:RightError>1.4999999999999982</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>19</d2p1:SampleCount>\r\n            <d2p1:SplitValue>13.5</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n        </d2p1:Nodes>\r\n      </d2p1:GBMTree>\r\n      <d2p1:GBMTree z:Id=\"28\">\r\n        <d2p1:Nodes z:Id=\"29\" z:Size=\"6\">\r\n          <d2p1:GBMNode z:Id=\"30\">\r\n            <d2p1:Depth>0</d2p1:Depth>\r\n            <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>0.70169938484575745</d2p1:LeftConstant>\r\n            <d2p1:LeftError>4.51195943689269</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>0.70169938484575745</d2p1:RightConstant>\r\n            <d2p1:RightError>4.51195943689269</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>26</d2p1:SampleCount>\r\n            <d2p1:SplitValue>-1</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"31\">\r\n            <d2p1:Depth>1</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>1.0178497808241769</d2p1:LeftConstant>\r\n            <d2p1:LeftError>3.5560561124147441</d2p1:LeftError>\r\n            <d2p1:LeftIndex>2</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.3940754481637947</d2p1:RightConstant>\r\n            <d2p1:RightError>5.5511151231257827E-16</d2p1:RightError>\r\n            <d2p1:RightIndex>3</d2p1:RightIndex>\r\n            <d2p1:SampleCount>26</d2p1:SampleCount>\r\n            <d2p1:SplitValue>20</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"32\">\r\n            <d2p1:Depth>2</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>-0.26129981339727465</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0.40445472313087061</d2p1:LeftError>\r\n            <d2p1:LeftIndex>4</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>1.2026374881990321</d2p1:RightConstant>\r\n            <d2p1:RightError>2.8223468437614274</d2p1:RightError>\r\n            <d2p1:RightIndex>5</d2p1:RightIndex>\r\n            <d2p1:SampleCount>22</d2p1:SampleCount>\r\n            <d2p1:SplitValue>4.5</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"33\">\r\n            <d2p1:Depth>2</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>-1.3940754481637925</d2p1:LeftConstant>\r\n            <d2p1:LeftError>-8.3266726846886741E-17</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.3940754481638016</d2p1:RightConstant>\r\n            <d2p1:RightError>5.6898930012039273E-16</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>4</d2p1:SampleCount>\r\n            <d2p1:SplitValue>25</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"34\">\r\n            <d2p1:Depth>3</d2p1:Depth>\r\n            <d2p1:FeatureIndex>0</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>1.8454997650794467</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.476363101919945</d2p1:RightConstant>\r\n            <d2p1:RightError>0.003125954915934176</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>3</d2p1:SampleCount>\r\n            <d2p1:SplitValue>4</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"35\">\r\n            <d2p1:Depth>3</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>1.5881207774717288</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0.92307692307692246</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>0.34571926201171</d2p1:RightConstant>\r\n            <d2p1:RightError>1.5000000000000016</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>19</d2p1:SampleCount>\r\n            <d2p1:SplitValue>13.5</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n        </d2p1:Nodes>\r\n      </d2p1:GBMTree>\r\n      <d2p1:GBMTree z:Id=\"36\">\r\n        <d2p1:Nodes z:Id=\"37\" z:Size=\"6\">\r\n          <d2p1:GBMNode z:Id=\"38\">\r\n            <d2p1:Depth>0</d2p1:Depth>\r\n            <d2p1:FeatureIndex>-1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>0.6437998722965409</d2p1:LeftConstant>\r\n            <d2p1:LeftError>4.1213084097097727</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>0.6437998722965409</d2p1:RightConstant>\r\n            <d2p1:RightError>4.1213084097097727</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>26</d2p1:SampleCount>\r\n            <d2p1:SplitValue>-1</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"39\">\r\n            <d2p1:Depth>1</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>0.92847183796929378</d2p1:LeftConstant>\r\n            <d2p1:LeftError>3.3439173220730334</d2p1:LeftError>\r\n            <d2p1:LeftIndex>2</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.342795767209559</d2p1:RightConstant>\r\n            <d2p1:RightError>-3.3306690738754696E-16</d2p1:RightError>\r\n            <d2p1:RightIndex>3</d2p1:RightIndex>\r\n            <d2p1:SampleCount>26</d2p1:SampleCount>\r\n            <d2p1:SplitValue>20</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"40\">\r\n            <d2p1:Depth>2</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>-0.24798179549281313</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0.32911873867457769</d2p1:LeftError>\r\n            <d2p1:LeftIndex>4</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>1.09231706285825</d2p1:RightConstant>\r\n            <d2p1:RightError>2.7471226162746536</d2p1:RightError>\r\n            <d2p1:RightIndex>5</d2p1:RightIndex>\r\n            <d2p1:SampleCount>22</d2p1:SampleCount>\r\n            <d2p1:SplitValue>4.5</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"41\">\r\n            <d2p1:Depth>2</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>-1.3427957672095576</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.3427957672095594</d2p1:RightConstant>\r\n            <d2p1:RightError>-3.3306690738754696E-16</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>4</d2p1:SampleCount>\r\n            <d2p1:SplitValue>23</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"42\">\r\n            <d2p1:Depth>3</d2p1:Depth>\r\n            <d2p1:FeatureIndex>0</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>1.7030147787421894</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>-1.411809703520875</d2p1:RightConstant>\r\n            <d2p1:RightError>0.0027812379638359475</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>3</d2p1:SampleCount>\r\n            <d2p1:SplitValue>4</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n          <d2p1:GBMNode z:Id=\"43\">\r\n            <d2p1:Depth>3</d2p1:Depth>\r\n            <d2p1:FeatureIndex>1</d2p1:FeatureIndex>\r\n            <d2p1:LeftConstant>1.4518067863692972</d2p1:LeftConstant>\r\n            <d2p1:LeftError>0.92307692307692313</d2p1:LeftError>\r\n            <d2p1:LeftIndex>-1</d2p1:LeftIndex>\r\n            <d2p1:RightConstant>0.309338924177296</d2p1:RightConstant>\r\n            <d2p1:RightError>1.4999999999999996</d2p1:RightError>\r\n            <d2p1:RightIndex>-1</d2p1:RightIndex>\r\n            <d2p1:SampleCount>19</d2p1:SampleCount>\r\n            <d2p1:SplitValue>13.5</d2p1:SplitValue>\r\n          </d2p1:GBMNode>\r\n        </d2p1:Nodes>\r\n      </d2p1:GBMTree>\r\n    </d2p1:ArrayOfGBMTree>\r\n  </Trees>\r\n</ClassificationGradientBoostModel>";
        
    }
}
