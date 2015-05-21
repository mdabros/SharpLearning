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
    public class GBMGradientBoostClassificationModelTest
    {
        [TestMethod]
        public void ClassificationGradientBoostModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new GBMGradientBoostClassificationLearner();
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

            var learner = new GBMGradientBoostClassificationLearner();
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

            var learner = new GBMGradientBoostClassificationLearner();
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

            var learner = new GBMGradientBoostClassificationLearner();
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

            var learner = new GBMGradientBoostClassificationLearner();
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

            var learner = new GBMGradientBoostClassificationLearner();
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 26.287331114005394, 46.265416757664667 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
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
    }
}
