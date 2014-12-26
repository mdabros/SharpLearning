using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.Metrics.Classification;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.Containers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

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

            var learner = new ClassificationMultinomialDevianceGradientBoostLearner();
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

            var learner = new ClassificationMultinomialDevianceGradientBoostLearner();
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

            var learner = new ClassificationMultinomialDevianceGradientBoostLearner();
            var sut = learner.Learn(observations, targets);

            var actual = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = sut.PredictProbability(observations.GetRow(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.913457002399443 }, { 1, 0.0865429976005569 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.560546939908104 }, { 1, 0.439453060091896 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.850920111318499 }, { 1, 0.149079888681501 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.920017138224685 }, { 1, 0.079982861775315 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.560546939908104 }, { 1, 0.439453060091896 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.908701638544693 }, { 1, 0.0912983614553073 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.333766307509478 }, { 1, 0.666233692490522 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.310654345298818 }, { 1, 0.689345654701182 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.868389272463763 }, { 1, 0.131610727536237 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.30500824511542 }, { 1, 0.69499175488458 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.107219943051265 }, { 1, 0.892780056948735 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.850920111318499 }, { 1, 0.149079888681501 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.762662663274623 }, { 1, 0.237337336725377 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.932188426982944 }, { 1, 0.0678115730170562 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.184743826917209 }, { 1, 0.815256173082791 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.911976727437457 }, { 1, 0.0880232725625425 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.908701638544693 }, { 1, 0.0912983614553073 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.750835408508563 }, { 1, 0.249164591491437 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.30500824511542 }, { 1, 0.69499175488458 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.124394439925571 }, { 1, 0.875605560074429 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.868389272463763 }, { 1, 0.131610727536237 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.30500824511542 }, { 1, 0.69499175488458 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.302049422791982 }, { 1, 0.697950577208018 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.911976727437457 }, { 1, 0.0880232725625425 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.908701638544693 }, { 1, 0.0912983614553073 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.868389272463763 }, { 1, 0.131610727536237 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationGradientBoostModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationMultinomialDevianceGradientBoostLearner();
            var sut = learner.Learn(observations, targets);

            var actual = sut.PredictProbability(observations);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.913457002399443 }, { 1, 0.0865429976005569 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.560546939908104 }, { 1, 0.439453060091896 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.850920111318499 }, { 1, 0.149079888681501 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.920017138224685 }, { 1, 0.079982861775315 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.560546939908104 }, { 1, 0.439453060091896 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.908701638544693 }, { 1, 0.0912983614553073 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.333766307509478 }, { 1, 0.666233692490522 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.310654345298818 }, { 1, 0.689345654701182 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.868389272463763 }, { 1, 0.131610727536237 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.30500824511542 }, { 1, 0.69499175488458 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.107219943051265 }, { 1, 0.892780056948735 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.850920111318499 }, { 1, 0.149079888681501 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.762662663274623 }, { 1, 0.237337336725377 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.932188426982944 }, { 1, 0.0678115730170562 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.184743826917209 }, { 1, 0.815256173082791 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.911976727437457 }, { 1, 0.0880232725625425 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.908701638544693 }, { 1, 0.0912983614553073 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.750835408508563 }, { 1, 0.249164591491437 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.30500824511542 }, { 1, 0.69499175488458 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.124394439925571 }, { 1, 0.875605560074429 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.868389272463763 }, { 1, 0.131610727536237 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.30500824511542 }, { 1, 0.69499175488458 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.302049422791982 }, { 1, 0.697950577208018 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.911976727437457 }, { 1, 0.0880232725625425 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.908701638544693 }, { 1, 0.0912983614553073 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.868389272463763 }, { 1, 0.131610727536237 }, }), };
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

            var learner = new ClassificationMultinomialDevianceGradientBoostLearner();
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, 
                { "AptitudeTestScore", 6.40056573424425 } };

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

            var learner = new ClassificationMultinomialDevianceGradientBoostLearner();
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 0.41075839873880021, 6.41753269622971 };

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
