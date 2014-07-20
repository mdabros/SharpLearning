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
        public void ClassificationRandomForestModel_Predict_Multiple()
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
        public void ClassificationRandomForestModel_Predict_Multiple_Indexed()
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
        public void ClassificationRandomForestModel_PredictProbability_Single()
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

            Assert.AreEqual(0.15384615384615385, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.664373451034092 }, { 1, 0.335626548965908 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.632080279164682 }, { 1, 0.367919720835318 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.742695498912254 }, { 1, 0.257304501087746 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.741555942562171 }, { 1, 0.258444057437828 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.632080279164682 }, { 1, 0.367919720835318 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.797452935919691 }, { 1, 0.202547064080309 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.507414467132754 }, { 1, 0.492585532867246 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.663136696603452 }, { 1, 0.336863303396548 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.822826039939854 }, { 1, 0.177173960060146 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.314370598355892 }, { 1, 0.685629401644107 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.449295935522278 }, { 1, 0.550704064477722 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.742695498912254 }, { 1, 0.257304501087746 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.668485262604612 }, { 1, 0.331514737395387 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.743232925864387 }, { 1, 0.256767074135613 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.411619906065727 }, { 1, 0.588380093934273 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.80709482670864 }, { 1, 0.192905173291359 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.787464840681596 }, { 1, 0.212535159318404 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.546511258023023 }, { 1, 0.453488741976977 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.323603931689226 }, { 1, 0.676396068310774 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.492891947570235 }, { 1, 0.507108052429765 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.733906369912598 }, { 1, 0.266093630087401 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.314370598355892 }, { 1, 0.685629401644107 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.523013567632918 }, { 1, 0.476986432367082 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.80709482670864 }, { 1, 0.192905173291359 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.797452935919691 }, { 1, 0.202547064080309 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.693053853964495 }, { 1, 0.306946146035505 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationRandomForestModel_PredictProbability_Multiple()
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

            Assert.AreEqual(0.15384615384615385, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.664373451034092 }, { 1, 0.335626548965908 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.632080279164682 }, { 1, 0.367919720835318 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.742695498912254 }, { 1, 0.257304501087746 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.741555942562171 }, { 1, 0.258444057437828 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.632080279164682 }, { 1, 0.367919720835318 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.797452935919691 }, { 1, 0.202547064080309 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.507414467132754 }, { 1, 0.492585532867246 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.663136696603452 }, { 1, 0.336863303396548 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.822826039939854 }, { 1, 0.177173960060146 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.314370598355892 }, { 1, 0.685629401644107 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.449295935522278 }, { 1, 0.550704064477722 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.742695498912254 }, { 1, 0.257304501087746 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.668485262604612 }, { 1, 0.331514737395387 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.743232925864387 }, { 1, 0.256767074135613 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.411619906065727 }, { 1, 0.588380093934273 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.80709482670864 }, { 1, 0.192905173291359 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.787464840681596 }, { 1, 0.212535159318404 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.546511258023023 }, { 1, 0.453488741976977 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.323603931689226 }, { 1, 0.676396068310774 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.492891947570235 }, { 1, 0.507108052429765 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.733906369912598 }, { 1, 0.266093630087401 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.314370598355892 }, { 1, 0.685629401644107 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.523013567632918 }, { 1, 0.476986432367082 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.80709482670864 }, { 1, 0.192905173291359 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.797452935919691 }, { 1, 0.202547064080309 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.693053853964495 }, { 1, 0.306946146035505 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationRandomForestModel_PredictProbability_Multiple_Indexed()
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

            Assert.AreEqual(0.3, error, 0.0000001);

            var expected = new ProbabilityPrediction[] {new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.664373451034092}, {1, 0.335626548965908}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.741555942562171}, {1, 0.258444057437828}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.632080279164682}, {1, 0.367919720835318}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.797452935919691}, {1, 0.202547064080309}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.507414467132754}, {1, 0.492585532867246}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.663136696603452}, {1, 0.336863303396548}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.822826039939854}, {1, 0.177173960060146}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.314370598355892}, {1, 0.685629401644107}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.733906369912598}, {1, 0.266093630087401}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.314370598355892}, {1, 0.685629401644107}, }),};
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationRandomForestModel_GetVariableImportance()
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
                { "AptitudeTestScore", 40.281512620464191 } };

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
            var expected = new double[] { 3.847968729029696, 9.5526917404656135 };

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
