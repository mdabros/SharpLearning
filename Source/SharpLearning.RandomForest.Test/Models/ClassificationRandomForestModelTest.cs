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
            
            Assert.AreEqual(0.26923076923076922, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.640341171771125 }, { 1, 0.359658828228875 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.650935423881244 }, { 1, 0.349064576118756 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.733778146389214 }, { 1, 0.266221853610785 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.632102355032308 }, { 1, 0.367897644967692 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.650935423881244 }, { 1, 0.349064576118756 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.729606065202427 }, { 1, 0.270393934797572 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.487576568855981 }, { 1, 0.512423431144019 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.699828120924483 }, { 1, 0.300171879075517 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.765651330394751 }, { 1, 0.234348669605248 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.44189058817 }, { 1, 0.55810941183 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.614494310299264 }, { 1, 0.385505689700736 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.733778146389214 }, { 1, 0.266221853610785 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.61462806620624 }, { 1, 0.38537193379376 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.735632387736103 }, { 1, 0.264367612263897 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.539394298340119 }, { 1, 0.460605701659881 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.749324346920709 }, { 1, 0.250675653079291 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.729332255678618 }, { 1, 0.270667744321382 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.485251172030584 }, { 1, 0.514748827969416 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.44189058817 }, { 1, 0.55810941183 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.614494310299264 }, { 1, 0.385505689700736 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.636274436219096 }, { 1, 0.363725563780904 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.44189058817 }, { 1, 0.55810941183 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.602473163918984 }, { 1, 0.397526836081016 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.749324346920709 }, { 1, 0.250675653079291 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.729606065202427 }, { 1, 0.270393934797572 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.644513252957912 }, { 1, 0.355486747042088 }, }), };
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

            Assert.AreEqual(0.26923076923076922, error, 0.0000001);

            var expected = new ProbabilityPrediction[] {new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.640341171771125}, {1, 0.359658828228875}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.650935423881244}, {1, 0.349064576118756}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.733778146389214}, {1, 0.266221853610785}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.632102355032308}, {1, 0.367897644967692}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.650935423881244}, {1, 0.349064576118756}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.729606065202427}, {1, 0.270393934797572}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.487576568855981}, {1, 0.512423431144019}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.699828120924483}, {1, 0.300171879075517}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.765651330394751}, {1, 0.234348669605248}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.44189058817}, {1, 0.55810941183}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.614494310299264}, {1, 0.385505689700736}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.733778146389214}, {1, 0.266221853610785}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.61462806620624}, {1, 0.38537193379376}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.735632387736103}, {1, 0.264367612263897}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.539394298340119}, {1, 0.460605701659881}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.749324346920709}, {1, 0.250675653079291}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.729332255678618}, {1, 0.270667744321382}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.485251172030584}, {1, 0.514748827969416}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.44189058817}, {1, 0.55810941183}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.614494310299264}, {1, 0.385505689700736}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.636274436219096}, {1, 0.363725563780904}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.44189058817}, {1, 0.55810941183}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.602473163918984}, {1, 0.397526836081016}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.749324346920709}, {1, 0.250675653079291}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.729606065202427}, {1, 0.270393934797572}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.644513252957912}, {1, 0.355486747042088}, }),};
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
