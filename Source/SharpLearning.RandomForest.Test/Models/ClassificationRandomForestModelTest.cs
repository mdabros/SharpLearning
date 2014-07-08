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

namespace SharpLearning.RandomForest.Test.Models
{
    [TestClass]
    public class ClassificationRandomForestModelTest
    {
        [TestMethod]
        public void ClassificationRandomForestModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.GetRow(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.26923076923076922, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationRandomForestModel_Precit_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.26923076923076922, error, 0.0000001);
        }


        [TestMethod]
        public void ClassificationRandomForestModel_PredictProbability_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = sut.PredictProbability(observations.GetRow(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.23076923076923078, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.37046763466152 }, { 0, 0.62953236533848 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.640331874137215 }, { 1, 0.359668125862785 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.742308446584375 }, { 1, 0.257691553415625 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.382225265919152 }, { 0, 0.617774734080848 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.640331874137215 }, { 1, 0.359668125862785 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.264731073836724 }, { 0, 0.735268926163276 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.560593414626309 }, { 0, 0.439406585373691 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.301990587714659 }, { 0, 0.698009412285341 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.781867734518663 }, { 1, 0.218132265481337 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.618914567157988 }, { 0, 0.381085432842012 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.403705729899615 }, { 0, 0.596294270100385 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.742308446584375 }, { 1, 0.257691553415625 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.40451776379795 }, { 0, 0.59548223620205 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.258409881140531 }, { 0, 0.741590118859468 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.501420375075561 }, { 0, 0.498579624924439 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.240642204372854 }, { 0, 0.759357795627145 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.264707264312914 }, { 0, 0.735292735687085 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.561571243814665 }, { 0, 0.438428756185335 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.618914567157988 }, { 0, 0.381085432842012 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.403705729899615 }, { 0, 0.596294270100385 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.624814254501948 }, { 1, 0.375185745498052 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.618914567157988 }, { 0, 0.381085432842012 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.419988182268368 }, { 0, 0.580011817731632 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.240642204372854 }, { 0, 0.759357795627145 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.264731073836724 }, { 0, 0.735268926163276 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.636571885759579 }, { 1, 0.363428114240421 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationRandomForestModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.PredictProbability(observations);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.23076923076923078, error, 0.0000001);
            Write(actual);
            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.37046763466152 }, { 0, 0.62953236533848 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.640331874137215 }, { 1, 0.359668125862785 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.742308446584375 }, { 1, 0.257691553415625 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.382225265919152 }, { 0, 0.617774734080848 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.640331874137215 }, { 1, 0.359668125862785 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.264731073836724 }, { 0, 0.735268926163276 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.560593414626309 }, { 0, 0.439406585373691 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.301990587714659 }, { 0, 0.698009412285341 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.781867734518663 }, { 1, 0.218132265481337 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.618914567157988 }, { 0, 0.381085432842012 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.403705729899615 }, { 0, 0.596294270100385 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.742308446584375 }, { 1, 0.257691553415625 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.40451776379795 }, { 0, 0.59548223620205 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.258409881140531 }, { 0, 0.741590118859468 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.501420375075561 }, { 0, 0.498579624924439 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.240642204372854 }, { 0, 0.759357795627145 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.264707264312914 }, { 0, 0.735292735687085 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.561571243814665 }, { 0, 0.438428756185335 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.618914567157988 }, { 0, 0.381085432842012 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.403705729899615 }, { 0, 0.596294270100385 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.624814254501948 }, { 1, 0.375185745498052 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.618914567157988 }, { 0, 0.381085432842012 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.419988182268368 }, { 0, 0.580011817731632 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.240642204372854 }, { 0, 0.759357795627145 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.264731073836724 }, { 0, 0.735268926163276 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.636571885759579 }, { 1, 0.363428114240421 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationRandomForestModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, { "PreviousExperience_month", 1 } };

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, { "AptitudeTestScore", 47.6329932736749 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void ClassificationRandomForestModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 3.7339437235781547, 7.8389861038646416 };

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
