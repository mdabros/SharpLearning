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
    public class ClassificationForestModelTest
    {
        [TestMethod]
        public void ClassificationForestModel_Predict_Single()
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

            Assert.AreEqual(0.23076923076923078, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationForestModel_Predict_Multiple()
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

            Assert.AreEqual(0.23076923076923078, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationForestModel_Predict_Multiple_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);
            var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };

            var predictions = sut.Predict(observations, indices);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var indexedTargets = targets.GetIndices(indices);
            var error = evaluator.Error(indexedTargets, predictions);

            Assert.AreEqual(0.2, error, 0.0000001);
        }


        [TestMethod]
        public void ClassificationForestModel_PredictProbability_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 1, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = sut.PredictProbability(observations.GetRow(i));
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

            var learner = new ClassificationRandomForestLearner(100, 1, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.PredictProbability(observations);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.076923076923076927, error, 0.0000001);
            
            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.659019957381799 }, { 1, 0.340980042618201 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.59688515242927 }, { 1, 0.40311484757073 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.719480409511369 }, { 1, 0.280519590488631 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.744205045986005 }, { 1, 0.255794954013994 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.59688515242927 }, { 1, 0.40311484757073 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.788828061859021 }, { 1, 0.211171938140978 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.495583888333888 }, { 1, 0.504416111666112 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.58936533014629 }, { 1, 0.41063466985371 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.830597486878446 }, { 1, 0.169402513121553 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.256507450882451 }, { 1, 0.743492549117549 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.349079795204795 }, { 1, 0.650920204795204 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.719480409511369 }, { 1, 0.280519590488631 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.681955044955045 }, { 1, 0.318044955044955 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.774851926882887 }, { 1, 0.225148073117113 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.304978632478632 }, { 1, 0.695021367521367 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.821320069851029 }, { 1, 0.17867993014897 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.788828061859021 }, { 1, 0.211171938140978 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.604202935952936 }, { 1, 0.395797064047064 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.277229673104673 }, { 1, 0.722770326895327 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.392208491070333 }, { 1, 0.607791508929667 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.717849096380056 }, { 1, 0.282150903619944 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.256507450882451 }, { 1, 0.743492549117549 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.452195415695416 }, { 1, 0.547804584304584 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.821320069851029 }, { 1, 0.17867993014897 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.788828061859021 }, { 1, 0.211171938140978 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.654524036635878 }, { 1, 0.345475963364121 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationForestModel_PredictProbability_Multiple_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 1, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };
            var actual = sut.PredictProbability(observations, indices);

            var indexedTargets = targets.GetIndices(indices);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(indexedTargets, actual.Select(p => p.Prediction).ToArray());
           
            Assert.AreEqual(0.2, error, 0.0000001);
            
            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.659019957381799 }, { 1, 0.340980042618201 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.744205045986005 }, { 1, 0.255794954013994 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.59688515242927 }, { 1, 0.40311484757073 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.788828061859021 }, { 1, 0.211171938140978 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.495583888333888 }, { 1, 0.504416111666112 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.58936533014629 }, { 1, 0.41063466985371 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.830597486878446 }, { 1, 0.169402513121553 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.256507450882451 }, { 1, 0.743492549117549 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.717849096380056 }, { 1, 0.282150903619944 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.256507450882451 }, { 1, 0.743492549117549 }, }), };
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

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, 
                { "AptitudeTestScore", 55.5555555555552 } };

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

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 1.4792899408283984, 2.6627218934911334 };

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
