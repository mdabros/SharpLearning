using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.DecisionTrees.Test.Models
{
    [TestClass]
    public class ClassificationCartModelTest
    {
        [TestMethod]
        public void ClassificationCartModel_Single_Observation()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var sut = new ClassificationCartLearner(1, 100, 0.001);
            var model = sut.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = model.Predict(observations.GetRow(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationCartModel_Single_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var sut = new ClassificationCartLearner(1, 100, 0.001);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationCartMode_PredictProbability_Single_Observation()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var sut = new ClassificationCartLearner(5, 100, 0.001);
            var model = sut.Learn(observations, targets);

            var actual = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = model.PredictProbability(observations.GetRow(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.23076923076923078, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.6 }, { 0, 0.4 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1 }, { 1, 0 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.6 }, { 0, 0.4 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.8 }, { 0, 0.2 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1 }, { 1, 0 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.8 }, { 0, 0.2 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1 }, { 1, 0 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.6 }, { 0, 0.4 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.6 }, { 0, 0.4 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.8 }, { 0, 0.2 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.8 }, { 0, 0.2 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1 }, { 1, 0 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.8 }, { 0, 0.2 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.6 }, { 0, 0.4 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1 }, { 1, 0 }, }) };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationCartMode_PredictProbability_Multiple_Observation()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var sut = new ClassificationCartLearner(5, 100, 0.001);
            var model = sut.Learn(observations, targets);

            var actual = model.PredictProbability(observations);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.23076923076923078, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.6 }, { 0, 0.4 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1 }, { 1, 0 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.6 }, { 0, 0.4 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.8 }, { 0, 0.2 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1 }, { 1, 0 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.8 }, { 0, 0.2 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1 }, { 1, 0 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.6 }, { 0, 0.4 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.6 }, { 0, 0.4 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.8 }, { 0, 0.2 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.8 }, { 0, 0.2 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1 }, { 1, 0 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.8 }, { 0, 0.2 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.6 }, { 0, 0.4 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.272727272727273 }, { 0, 0.727272727272727 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1 }, { 1, 0 }, }) };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationCartModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, { "PreviousExperience_month", 1 } };

            var sut = new ClassificationCartLearner(1, 100, 0.001);
            var model = sut.Learn(observations, targets);
            
            var actual = model.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, { "AptitudeTestScore", 17.2872340425532 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void ClassificationCartModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new ClassificationCartLearner(1, 100, 0.001);
            var model = sut.Learn(observations, targets);

            var actual = model.GetRawVariableImportance();
            var expected = new double[] { 0.064102564102564111, 0.37080867850098614 };

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
