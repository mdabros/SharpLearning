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

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.33837148154563 }, { 0, 0.66162851845437 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.354415359494307 }, { 0, 0.645584640505693 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.234541059774419 }, { 0, 0.765458940225581 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.336094786018935 }, { 0, 0.663905213981065 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.354415359494307 }, { 0, 0.645584640505693 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.237433958542318 }, { 0, 0.762566041457682 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.535691516086253 }, { 0, 0.464308483913747 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.277875836109195 }, { 0, 0.722124163890804 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.213201589303369 }, { 0, 0.78679841069663 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.627916662285083 }, { 0, 0.372083337714917 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.381942910117059 }, { 0, 0.618057089882941 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.234541059774419 }, { 0, 0.765458940225581 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.43374969767075 }, { 0, 0.56625030232925 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.259665916642697 }, { 0, 0.740334083357303 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.529255834808466 }, { 0, 0.470744165191534 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.234557164283944 }, { 0, 0.765442835716055 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.237433958542318 }, { 0, 0.762566041457682 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.553749995618417 }, { 0, 0.446250004381583 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.627916662285083 }, { 0, 0.372083337714917 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.381942910117059 }, { 0, 0.618057089882941 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.333201887251036 }, { 0, 0.666798112748964 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.627916662285083 }, { 0, 0.372083337714917 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.452212373883426 }, { 0, 0.547787626116573 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.234557164283944 }, { 0, 0.765442835716055 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.237433958542318 }, { 0, 0.762566041457682 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.335478582777731 }, { 0, 0.664521417222268 }, }), };
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
            
            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.33837148154563 }, { 0, 0.66162851845437 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.354415359494307 }, { 0, 0.645584640505693 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.234541059774419 }, { 0, 0.765458940225581 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.336094786018935 }, { 0, 0.663905213981065 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.354415359494307 }, { 0, 0.645584640505693 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.237433958542318 }, { 0, 0.762566041457682 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.535691516086253 }, { 0, 0.464308483913747 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.277875836109195 }, { 0, 0.722124163890804 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.213201589303369 }, { 0, 0.78679841069663 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.627916662285083 }, { 0, 0.372083337714917 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.381942910117059 }, { 0, 0.618057089882941 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.234541059774419 }, { 0, 0.765458940225581 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.43374969767075 }, { 0, 0.56625030232925 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.259665916642697 }, { 0, 0.740334083357303 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.529255834808466 }, { 0, 0.470744165191534 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.234557164283944 }, { 0, 0.765442835716055 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.237433958542318 }, { 0, 0.762566041457682 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.553749995618417 }, { 0, 0.446250004381583 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.627916662285083 }, { 0, 0.372083337714917 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.381942910117059 }, { 0, 0.618057089882941 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.333201887251036 }, { 0, 0.666798112748964 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 1, 0.627916662285083 }, { 0, 0.372083337714917 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.452212373883426 }, { 0, 0.547787626116573 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.234557164283944 }, { 0, 0.765442835716055 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.237433958542318 }, { 0, 0.762566041457682 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 1, 0.335478582777731 }, { 0, 0.664521417222268 }, }), };
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
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, { "AptitudeTestScore", 32.6860468264474 } };

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
            var expected = new double[] { 3.1810336312360561, 9.73208429923124 };

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
