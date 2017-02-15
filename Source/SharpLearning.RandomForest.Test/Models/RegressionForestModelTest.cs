using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Models;
using SharpLearning.RandomForest.Test.Properties;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.RandomForest.Test.Models
{
    [TestClass]
    public class RegressionForestModelTest
    {
        [TestMethod]
        public void RegressionForestModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.Row(i));
            }

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.15392316626859898, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionForestModel_Predict_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.15392316626859898, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionForestModel_PredictCertainty_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = new CertaintyPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = sut.PredictCertainty(observations.Row(i));
            }

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.15392316626859898, error, 0.0000001);

            var expected = new CertaintyPrediction[] { new CertaintyPrediction(0.392351481851482, 0.0438279118815934), new CertaintyPrediction(0.386089490574785, 0.0667804041796964), new CertaintyPrediction(0.243836965322259, 0.0425186414889028), new CertaintyPrediction(0.329400543900544, 0.0573826198539963), new CertaintyPrediction(0.386089490574785, 0.0667804041796964), new CertaintyPrediction(0.200186171671466, 0.0375041708478056), new CertaintyPrediction(0.583565046065046, 0.0595672285510552), new CertaintyPrediction(0.248448797934092, 0.0449797371029248), new CertaintyPrediction(0.182894685379979, 0.0315534346029886), new CertaintyPrediction(0.682394411144411, 0.0615061393123617), new CertaintyPrediction(0.44569671994672, 0.0445745111801124), new CertaintyPrediction(0.243836965322259, 0.0425186414889028), new CertaintyPrediction(0.468528163013457, 0.0726477666545773), new CertaintyPrediction(0.211520587255881, 0.0551497857873964), new CertaintyPrediction(0.592137109622404, 0.0794360199970154), new CertaintyPrediction(0.153096705582, 0.0304859048440341), new CertaintyPrediction(0.162090933576228, 0.0341737062001491), new CertaintyPrediction(0.599291236541237, 0.0531997732586624), new CertaintyPrediction(0.680394411144411, 0.0606317169569394), new CertaintyPrediction(0.44569671994672, 0.0445745111801124), new CertaintyPrediction(0.300534743034743, 0.0603082500896416), new CertaintyPrediction(0.682394411144411, 0.0615061393123617), new CertaintyPrediction(0.5059524054377, 0.0653003480131079), new CertaintyPrediction(0.186691943677238, 0.0343771214986317), new CertaintyPrediction(0.166090933576228, 0.0344609787315392), new CertaintyPrediction(0.35805710955711, 0.0525006655402324), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void RegressionForestModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);
            var actual = sut.PredictCertainty(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.15392316626859898, error, 0.0000001);

            Write(actual);

            var expected = new CertaintyPrediction[] { new CertaintyPrediction(0.392351481851482, 0.0438279118815934), new CertaintyPrediction(0.386089490574785, 0.0667804041796964), new CertaintyPrediction(0.243836965322259, 0.0425186414889028), new CertaintyPrediction(0.329400543900544, 0.0573826198539963), new CertaintyPrediction(0.386089490574785, 0.0667804041796964), new CertaintyPrediction(0.200186171671466, 0.0375041708478056), new CertaintyPrediction(0.583565046065046, 0.0595672285510552), new CertaintyPrediction(0.248448797934092, 0.0449797371029248), new CertaintyPrediction(0.182894685379979, 0.0315534346029886), new CertaintyPrediction(0.682394411144411, 0.0615061393123617), new CertaintyPrediction(0.44569671994672, 0.0445745111801124), new CertaintyPrediction(0.243836965322259, 0.0425186414889028), new CertaintyPrediction(0.468528163013457, 0.0726477666545773), new CertaintyPrediction(0.211520587255881, 0.0551497857873964), new CertaintyPrediction(0.592137109622404, 0.0794360199970154), new CertaintyPrediction(0.153096705582, 0.0304859048440341), new CertaintyPrediction(0.162090933576228, 0.0341737062001491), new CertaintyPrediction(0.599291236541237, 0.0531997732586624), new CertaintyPrediction(0.680394411144411, 0.0606317169569394), new CertaintyPrediction(0.44569671994672, 0.0445745111801124), new CertaintyPrediction(0.300534743034743, 0.0603082500896416), new CertaintyPrediction(0.682394411144411, 0.0615061393123617), new CertaintyPrediction(0.5059524054377, 0.0653003480131079), new CertaintyPrediction(0.186691943677238, 0.0343771214986317), new CertaintyPrediction(0.166090933576228, 0.0344609787315392), new CertaintyPrediction(0.35805710955711, 0.0525006655402324), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void RegressionForestModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learner = new RegressionRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, 
                { "AptitudeTestScore", 41.8626323997622 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void RegressionForestModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new RegressionRandomForestLearner(100, 5, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 50.73913687047915, 121.20388509244185 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        [TestMethod]
        public void RegressionForestModel_Save()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new RegressionRandomForestLearner(2, 5, 100, 1, 0.0001, 1.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var writer = new StringWriter();
            sut.Save(() => writer);
   
            var actual = writer.ToString();
            Assert.AreEqual(ClassificationForestModelString, actual);
        }

        [TestMethod]
        public void RegressionForestModel_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var reader = new StringReader(ClassificationForestModelString);
            var sut = RegressionForestModel.Load(() => reader);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.19015066282923429, error, 0.0000001);
        }

        void Write(CertaintyPrediction[] predictions)
        {
            var value = "new CertaintyPrediction[] {";
            foreach (var item in predictions)
            {
                value += "new CertaintyPrediction(" + item.Prediction + ", " + item.Variance + "), ";
            }

            value += "};";

            Trace.WriteLine(value);
        }

        readonly string ClassificationForestModelString =
            "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<RegressionForestModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.RandomForest.Models\">\r\n  <m_models xmlns:d2p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Models\" z:Id=\"2\" z:Size=\"2\">\r\n    <d2p1:RegressionDecisionTreeModel z:Id=\"3\">\r\n      <d2p1:Tree xmlns:d4p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\" z:Id=\"4\">\r\n        <d4p1:Nodes z:Id=\"5\" z:Size=\"7\">\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>0</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>4</d4p1:RightIndex>\r\n            <d4p1:Value>9.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>2</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>1</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>3</d4p1:RightIndex>\r\n            <d4p1:Value>4.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>0</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>2</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0.66666666666666663</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>3</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>0</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>5</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>4</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>6</d4p1:RightIndex>\r\n            <d4p1:Value>3</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>2</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>5</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>1</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>3</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>6</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0.16666666666666666</d4p1:Value>\r\n          </d4p1:Node>\r\n        </d4p1:Nodes>\r\n        <d4p1:Probabilities xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"6\" z:Size=\"4\">\r\n          <d5p1:ArrayOfdouble z:Id=\"7\" z:Size=\"0\" />\r\n          <d5p1:ArrayOfdouble z:Id=\"8\" z:Size=\"0\" />\r\n          <d5p1:ArrayOfdouble z:Id=\"9\" z:Size=\"0\" />\r\n          <d5p1:ArrayOfdouble z:Id=\"10\" z:Size=\"0\" />\r\n        </d4p1:Probabilities>\r\n        <d4p1:TargetNames xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"11\" z:Size=\"2\">\r\n          <d5p1:double>0</d5p1:double>\r\n          <d5p1:double>1</d5p1:double>\r\n        </d4p1:TargetNames>\r\n        <d4p1:VariableImportance xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"12\" z:Size=\"2\">\r\n          <d5p1:double>1.1217948717948718</d5p1:double>\r\n          <d5p1:double>1.3333333333333333</d5p1:double>\r\n        </d4p1:VariableImportance>\r\n      </d2p1:Tree>\r\n      <d2p1:m_variableImportance xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"12\" i:nil=\"true\" />\r\n    </d2p1:RegressionDecisionTreeModel>\r\n    <d2p1:RegressionDecisionTreeModel z:Id=\"13\">\r\n      <d2p1:Tree xmlns:d4p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\" z:Id=\"14\">\r\n        <d4p1:Nodes z:Id=\"15\" z:Size=\"7\">\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>0</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>0</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>4</d4p1:RightIndex>\r\n            <d4p1:Value>3.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>2</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>1</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>3</d4p1:RightIndex>\r\n            <d4p1:Value>13.5</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>0</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>2</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>3</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0.42857142857142855</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>-1</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>5</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>4</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>6</d4p1:RightIndex>\r\n            <d4p1:Value>17</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>2</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>5</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0.6</d4p1:Value>\r\n          </d4p1:Node>\r\n          <d4p1:Node>\r\n            <d4p1:FeatureIndex>-1</d4p1:FeatureIndex>\r\n            <d4p1:LeafProbabilityIndex>3</d4p1:LeafProbabilityIndex>\r\n            <d4p1:LeftIndex>-1</d4p1:LeftIndex>\r\n            <d4p1:NodeIndex>6</d4p1:NodeIndex>\r\n            <d4p1:RightIndex>-1</d4p1:RightIndex>\r\n            <d4p1:Value>0.5</d4p1:Value>\r\n          </d4p1:Node>\r\n        </d4p1:Nodes>\r\n        <d4p1:Probabilities xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"16\" z:Size=\"4\">\r\n          <d5p1:ArrayOfdouble z:Id=\"17\" z:Size=\"0\" />\r\n          <d5p1:ArrayOfdouble z:Id=\"18\" z:Size=\"0\" />\r\n          <d5p1:ArrayOfdouble z:Id=\"19\" z:Size=\"0\" />\r\n          <d5p1:ArrayOfdouble z:Id=\"20\" z:Size=\"0\" />\r\n        </d4p1:Probabilities>\r\n        <d4p1:TargetNames xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"21\" z:Size=\"2\">\r\n          <d5p1:double>0</d5p1:double>\r\n          <d5p1:double>1</d5p1:double>\r\n        </d4p1:TargetNames>\r\n        <d4p1:VariableImportance xmlns:d5p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"22\" z:Size=\"2\">\r\n          <d5p1:double>0.757342657342657</d5p1:double>\r\n          <d5p1:double>0.40714285714285714</d5p1:double>\r\n        </d4p1:VariableImportance>\r\n      </d2p1:Tree>\r\n      <d2p1:m_variableImportance xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Ref=\"22\" i:nil=\"true\" />\r\n    </d2p1:RegressionDecisionTreeModel>\r\n  </m_models>\r\n  <m_rawVariableImportance xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"23\" z:Size=\"2\">\r\n    <d2p1:double>1.879137529137529</d2p1:double>\r\n    <d2p1:double>1.7404761904761905</d2p1:double>\r\n  </m_rawVariableImportance>\r\n</RegressionForestModel>";

    }
}
