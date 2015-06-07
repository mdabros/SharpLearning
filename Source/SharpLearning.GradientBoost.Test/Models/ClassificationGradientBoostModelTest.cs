using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using System.Linq;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.Metrics.Classification;
using SharpLearning.GradientBoost.GBM;
using SharpLearning.Containers;
using System.Collections.Generic;
using System.Diagnostics;

namespace SharpLearning.GradientBoost.Test.GBM
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
                predictions[i] = sut.Predict(observations.GetRow(i));
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
                actual[i] = sut.PredictProbability(observations.GetRow(i));
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
        public void GBMGradientBoostClassificationModel_Save()
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
        public void GBMGradientBoostClassificationModel_Load()
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
        "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<ClassificationGradientBoostModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.GradientBoost.GBM\">\r\n  <m_featureCount>2</m_featureCount>\r\n  <m_initialLoss>-0.47000362924573558</m_initialLoss>\r\n  <m_learningRate>0.1</m_learningRate>\r\n  <m_targetNames xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"2\" z:Size=\"2\">\r\n    <d2p1:double>0</d2p1:double>\r\n    <d2p1:double>1</d2p1:double>\r\n  </m_targetNames>\r\n  <m_trees z:Id=\"3\" z:Size=\"1\">\r\n    <ArrayOfGBMTree z:Id=\"4\" z:Size=\"5\">\r\n      <GBMTree z:Id=\"5\">\r\n        <m_nodes z:Id=\"6\" z:Size=\"5\">\r\n          <GBMNode z:Id=\"7\">\r\n            <Depth>0</Depth>\r\n            <FeatureIndex>-1</FeatureIndex>\r\n            <LeftConstant>0.9749999999999992</LeftConstant>\r\n            <LeftError>6.1538461538461577</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>0.9749999999999992</RightConstant>\r\n            <RightError>6.1538461538461577</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>26</SampleCount>\r\n            <SplitValue>-1</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"8\">\r\n            <Depth>1</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>1.4477272727272721</LeftConstant>\r\n            <LeftError>4.3636363636363686</LeftError>\r\n            <LeftIndex>2</LeftIndex>\r\n            <RightConstant>-1.625000000000002</RightConstant>\r\n            <RightError>-1.5543122344752192E-15</RightError>\r\n            <RightIndex>3</RightIndex>\r\n            <SampleCount>26</SampleCount>\r\n            <SplitValue>20</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"9\">\r\n            <Depth>2</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>-1.625</LeftConstant>\r\n            <LeftError>0</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>1.5940476190476185</RightConstant>\r\n            <RightError>3.8095238095238142</RightError>\r\n            <RightIndex>4</RightIndex>\r\n            <SampleCount>22</SampleCount>\r\n            <SplitValue>2.5</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"10\">\r\n            <Depth>2</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>-1.625</LeftConstant>\r\n            <LeftError>0</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>-1.6250000000000027</RightConstant>\r\n            <RightError>-1.609823385706477E-15</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>4</SampleCount>\r\n            <SplitValue>23</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"11\">\r\n            <Depth>3</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>2.0366666666666666</LeftConstant>\r\n            <LeftError>1.7333333333333374</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>0.4874999999999996</RightConstant>\r\n            <RightError>1.5</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>21</SampleCount>\r\n            <SplitValue>13.5</SplitValue>\r\n          </GBMNode>\r\n        </m_nodes>\r\n      </GBMTree>\r\n      <GBMTree z:Id=\"12\">\r\n        <m_nodes z:Id=\"13\" z:Size=\"6\">\r\n          <GBMNode z:Id=\"14\">\r\n            <Depth>0</Depth>\r\n            <FeatureIndex>-1</FeatureIndex>\r\n            <LeftConstant>0.86059480408450362</LeftConstant>\r\n            <LeftError>5.596711778030242</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>0.86059480408450362</RightConstant>\r\n            <RightError>5.596711778030242</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>26</SampleCount>\r\n            <SplitValue>-1</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"15\">\r\n            <Depth>1</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>1.266063090014917</LeftConstant>\r\n            <LeftError>4.1463589231846409</LeftError>\r\n            <LeftIndex>2</LeftIndex>\r\n            <RightConstant>-1.5312600563908738</RightConstant>\r\n            <RightError>-1.27675647831893E-15</RightError>\r\n            <RightIndex>3</RightIndex>\r\n            <SampleCount>26</SampleCount>\r\n            <SplitValue>20</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"16\">\r\n            <Depth>2</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>-0.2989037549136308</LeftConstant>\r\n            <LeftError>0.6137878335116782</LeftError>\r\n            <LeftIndex>4</LeftIndex>\r\n            <RightConstant>1.5087703239250865</RightConstant>\r\n            <RightError>3.0331204175116282</RightError>\r\n            <RightIndex>5</RightIndex>\r\n            <SampleCount>22</SampleCount>\r\n            <SplitValue>4.5</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"17\">\r\n            <Depth>2</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>-1.5312600563908738</LeftConstant>\r\n            <LeftError>0</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>-1.5312600563908738</RightConstant>\r\n            <RightError>-1.27675647831893E-15</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>4</SampleCount>\r\n            <SplitValue>23</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"18\">\r\n            <Depth>3</Depth>\r\n            <FeatureIndex>0</FeatureIndex>\r\n            <LeftConstant>2.3051747796578996</LeftConstant>\r\n            <LeftError>0</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>-1.6534579988148759</RightConstant>\r\n            <RightError>0.0037726356416253326</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>3</SampleCount>\r\n            <SplitValue>4</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"19\">\r\n            <Depth>3</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>1.9919935727839668</LeftConstant>\r\n            <LeftError>0.92307692307692335</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>0.43382354239469473</RightConstant>\r\n            <RightError>1.4999999999999982</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>19</SampleCount>\r\n            <SplitValue>13.5</SplitValue>\r\n          </GBMNode>\r\n        </m_nodes>\r\n      </GBMTree>\r\n      <GBMTree z:Id=\"20\">\r\n        <m_nodes z:Id=\"21\" z:Size=\"6\">\r\n          <GBMNode z:Id=\"22\">\r\n            <Depth>0</Depth>\r\n            <FeatureIndex>-1</FeatureIndex>\r\n            <LeftConstant>0.772024071047949</LeftConstant>\r\n            <LeftError>4.9956325710355829</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>0.772024071047949</RightConstant>\r\n            <RightError>4.9956325710355829</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>26</SampleCount>\r\n            <SplitValue>-1</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"23\">\r\n            <Depth>1</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>1.1271620452174398</LeftConstant>\r\n            <LeftError>3.81901068866107</LeftError>\r\n            <LeftIndex>2</LeftIndex>\r\n            <RightConstant>-1.4558326033028339</RightConstant>\r\n            <RightError>-5.5511151231257827E-17</RightError>\r\n            <RightIndex>3</RightIndex>\r\n            <SampleCount>26</SampleCount>\r\n            <SplitValue>20</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"24\">\r\n            <Depth>2</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>-0.27786210243575493</LeftConstant>\r\n            <LeftError>0.49774616548974571</LeftError>\r\n            <LeftIndex>4</LeftIndex>\r\n            <RightConstant>1.3374975310021555</RightConstant>\r\n            <RightError>2.9159510952504579</RightError>\r\n            <RightIndex>5</RightIndex>\r\n            <SampleCount>22</SampleCount>\r\n            <SplitValue>4.5</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"25\">\r\n            <Depth>2</Depth>\r\n            <FeatureIndex>0</FeatureIndex>\r\n            <LeftConstant>-1.4558326033028344</LeftConstant>\r\n            <LeftError>0</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>-1.4558326033028339</RightConstant>\r\n            <RightError>-1.1102230246251565E-16</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>4</SampleCount>\r\n            <SplitValue>2.5</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"26\">\r\n            <Depth>3</Depth>\r\n            <FeatureIndex>0</FeatureIndex>\r\n            <LeftConstant>2.0364687310490139</LeftConstant>\r\n            <LeftError>0</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>-1.5552695523522238</RightConstant>\r\n            <RightError>0.0034643522773981916</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>3</SampleCount>\r\n            <SplitValue>4</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"27\">\r\n            <Depth>3</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>1.7614034635323572</LeftConstant>\r\n            <LeftError>0.923076923076926</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>0.38692819292266051</RightConstant>\r\n            <RightError>1.4999999999999982</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>19</SampleCount>\r\n            <SplitValue>13.5</SplitValue>\r\n          </GBMNode>\r\n        </m_nodes>\r\n      </GBMTree>\r\n      <GBMTree z:Id=\"28\">\r\n        <m_nodes z:Id=\"29\" z:Size=\"6\">\r\n          <GBMNode z:Id=\"30\">\r\n            <Depth>0</Depth>\r\n            <FeatureIndex>-1</FeatureIndex>\r\n            <LeftConstant>0.70169938484575745</LeftConstant>\r\n            <LeftError>4.51195943689269</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>0.70169938484575745</RightConstant>\r\n            <RightError>4.51195943689269</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>26</SampleCount>\r\n            <SplitValue>-1</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"31\">\r\n            <Depth>1</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>1.0178497808241769</LeftConstant>\r\n            <LeftError>3.5560561124147441</LeftError>\r\n            <LeftIndex>2</LeftIndex>\r\n            <RightConstant>-1.3940754481637947</RightConstant>\r\n            <RightError>5.5511151231257827E-16</RightError>\r\n            <RightIndex>3</RightIndex>\r\n            <SampleCount>26</SampleCount>\r\n            <SplitValue>20</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"32\">\r\n            <Depth>2</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>-0.26129981339727465</LeftConstant>\r\n            <LeftError>0.40445472313087061</LeftError>\r\n            <LeftIndex>4</LeftIndex>\r\n            <RightConstant>1.2026374881990321</RightConstant>\r\n            <RightError>2.8223468437614274</RightError>\r\n            <RightIndex>5</RightIndex>\r\n            <SampleCount>22</SampleCount>\r\n            <SplitValue>4.5</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"33\">\r\n            <Depth>2</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>-1.3940754481637925</LeftConstant>\r\n            <LeftError>-8.3266726846886741E-17</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>-1.3940754481638016</RightConstant>\r\n            <RightError>5.6898930012039273E-16</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>4</SampleCount>\r\n            <SplitValue>25</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"34\">\r\n            <Depth>3</Depth>\r\n            <FeatureIndex>0</FeatureIndex>\r\n            <LeftConstant>1.8454997650794467</LeftConstant>\r\n            <LeftError>0</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>-1.476363101919945</RightConstant>\r\n            <RightError>0.003125954915934176</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>3</SampleCount>\r\n            <SplitValue>4</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"35\">\r\n            <Depth>3</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>1.5881207774717288</LeftConstant>\r\n            <LeftError>0.92307692307692246</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>0.34571926201171</RightConstant>\r\n            <RightError>1.5000000000000016</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>19</SampleCount>\r\n            <SplitValue>13.5</SplitValue>\r\n          </GBMNode>\r\n        </m_nodes>\r\n      </GBMTree>\r\n      <GBMTree z:Id=\"36\">\r\n        <m_nodes z:Id=\"37\" z:Size=\"6\">\r\n          <GBMNode z:Id=\"38\">\r\n            <Depth>0</Depth>\r\n            <FeatureIndex>-1</FeatureIndex>\r\n            <LeftConstant>0.6437998722965409</LeftConstant>\r\n            <LeftError>4.1213084097097727</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>0.6437998722965409</RightConstant>\r\n            <RightError>4.1213084097097727</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>26</SampleCount>\r\n            <SplitValue>-1</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"39\">\r\n            <Depth>1</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>0.92847183796929378</LeftConstant>\r\n            <LeftError>3.3439173220730334</LeftError>\r\n            <LeftIndex>2</LeftIndex>\r\n            <RightConstant>-1.342795767209559</RightConstant>\r\n            <RightError>-3.3306690738754696E-16</RightError>\r\n            <RightIndex>3</RightIndex>\r\n            <SampleCount>26</SampleCount>\r\n            <SplitValue>20</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"40\">\r\n            <Depth>2</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>-0.24798179549281313</LeftConstant>\r\n            <LeftError>0.32911873867457769</LeftError>\r\n            <LeftIndex>4</LeftIndex>\r\n            <RightConstant>1.09231706285825</RightConstant>\r\n            <RightError>2.7471226162746536</RightError>\r\n            <RightIndex>5</RightIndex>\r\n            <SampleCount>22</SampleCount>\r\n            <SplitValue>4.5</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"41\">\r\n            <Depth>2</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>-1.3427957672095576</LeftConstant>\r\n            <LeftError>0</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>-1.3427957672095594</RightConstant>\r\n            <RightError>-3.3306690738754696E-16</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>4</SampleCount>\r\n            <SplitValue>23</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"42\">\r\n            <Depth>3</Depth>\r\n            <FeatureIndex>0</FeatureIndex>\r\n            <LeftConstant>1.7030147787421894</LeftConstant>\r\n            <LeftError>0</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>-1.411809703520875</RightConstant>\r\n            <RightError>0.0027812379638359475</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>3</SampleCount>\r\n            <SplitValue>4</SplitValue>\r\n          </GBMNode>\r\n          <GBMNode z:Id=\"43\">\r\n            <Depth>3</Depth>\r\n            <FeatureIndex>1</FeatureIndex>\r\n            <LeftConstant>1.4518067863692972</LeftConstant>\r\n            <LeftError>0.92307692307692313</LeftError>\r\n            <LeftIndex>-1</LeftIndex>\r\n            <RightConstant>0.309338924177296</RightConstant>\r\n            <RightError>1.4999999999999996</RightError>\r\n            <RightIndex>-1</RightIndex>\r\n            <SampleCount>19</SampleCount>\r\n            <SplitValue>13.5</SplitValue>\r\n          </GBMNode>\r\n        </m_nodes>\r\n      </GBMTree>\r\n    </ArrayOfGBMTree>\r\n  </m_trees>\r\n</ClassificationGradientBoostModel>";
    }
}
