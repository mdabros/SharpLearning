using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.AdaBoost.Test.Properties;
using SharpLearning.Containers;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.AdaBoost.Test.Models
{
    [TestClass]
    public class ClassificationAdaBoostModelTest
    {
        [TestMethod]
        public void ClassificationAdaBoostModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationAdaBoostLearner(10);
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
        public void ClassificationAdaBoostModel_Precit_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationAdaBoostLearner(10);
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationAdaBoostModel_Precit_Multiple_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationAdaBoostLearner(10);
            var sut = learner.Learn(observations, targets);
            var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };

            var predictions = sut.Predict(observations, indices);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var indexedTargets = targets.GetIndices(indices);
            var error = evaluator.Error(indexedTargets, predictions);

            Assert.AreEqual(0.0, error, 0.0000001);
        }


        [TestMethod]
        public void ClassificationAdaBoostModel_PredictProbability_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationAdaBoostLearner(10, 1, 3);
            var sut = learner.Learn(observations, targets);

            var actual = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = sut.PredictProbability(observations.GetRow(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);

            var expected = new ProbabilityPrediction[] {new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.553917222019051}, {1, 0.446082777980949}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.455270122123639}, {1, 0.544729877876361}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.590671208378385}, {1, 0.409328791621616}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.564961572849738}, {1, 0.435038427150263}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.455270122123639}, {1, 0.544729877876361}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.549970403132686}, {1, 0.450029596867314}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.417527839140627}, {1, 0.582472160859373}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.409988559960094}, {1, 0.590011440039906}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.630894242807786}, {1, 0.369105757192214}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.436954866525023}, {1, 0.563045133474978}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.461264944069783}, {1, 0.538735055930217}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.590671208378385}, {1, 0.409328791621616}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.549503146925505}, {1, 0.450496853074495}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.537653803214063}, {1, 0.462346196785938}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.37650723540928}, {1, 0.62349276459072}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.573579890413618}, {1, 0.426420109586382}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.549970403132686}, {1, 0.450029596867314}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.524371409810479}, {1, 0.475628590189522}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.436954866525023}, {1, 0.563045133474978}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.471117379964633}, {1, 0.528882620035367}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.630894242807786}, {1, 0.369105757192214}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.436954866525023}, {1, 0.563045133474978}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.404976804073458}, {1, 0.595023195926542}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.573579890413618}, {1, 0.426420109586382}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.549970403132686}, {1, 0.450029596867314}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.630894242807786}, {1, 0.369105757192214}, }),};
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationAdaBoostModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationAdaBoostLearner(10, 1, 3);
            var sut = learner.Learn(observations, targets);

            var actual = sut.PredictProbability(observations);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.553917222019051 }, { 1, 0.446082777980949 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.455270122123639 }, { 1, 0.544729877876361 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.590671208378385 }, { 1, 0.409328791621616 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.564961572849738 }, { 1, 0.435038427150263 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.455270122123639 }, { 1, 0.544729877876361 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.549970403132686 }, { 1, 0.450029596867314 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.417527839140627 }, { 1, 0.582472160859373 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.409988559960094 }, { 1, 0.590011440039906 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.630894242807786 }, { 1, 0.369105757192214 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.436954866525023 }, { 1, 0.563045133474978 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.461264944069783 }, { 1, 0.538735055930217 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.590671208378385 }, { 1, 0.409328791621616 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.549503146925505 }, { 1, 0.450496853074495 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.537653803214063 }, { 1, 0.462346196785938 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.37650723540928 }, { 1, 0.62349276459072 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.573579890413618 }, { 1, 0.426420109586382 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.549970403132686 }, { 1, 0.450029596867314 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.524371409810479 }, { 1, 0.475628590189522 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.436954866525023 }, { 1, 0.563045133474978 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.471117379964633 }, { 1, 0.528882620035367 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.630894242807786 }, { 1, 0.369105757192214 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.436954866525023 }, { 1, 0.563045133474978 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.404976804073458 }, { 1, 0.595023195926542 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.573579890413618 }, { 1, 0.426420109586382 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.549970403132686 }, { 1, 0.450029596867314 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.630894242807786 }, { 1, 0.369105757192214 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationAdaBoostModel_PredictProbability_Multiple_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationAdaBoostLearner(10, 1, 3);
            var sut = learner.Learn(observations, targets);

            var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };
            var actual = sut.PredictProbability(observations, indices);

            var indexedTargets = targets.GetIndices(indices);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(indexedTargets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.0, error, 0.0000001);
            
            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.553917222019051 }, { 1, 0.446082777980949 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.564961572849738 }, { 1, 0.435038427150263 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.455270122123639 }, { 1, 0.544729877876361 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.549970403132686 }, { 1, 0.450029596867314 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.417527839140627 }, { 1, 0.582472160859373 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.409988559960094 }, { 1, 0.590011440039906 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.630894242807786 }, { 1, 0.369105757192214 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.436954866525023 }, { 1, 0.563045133474978 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.630894242807786 }, { 1, 0.369105757192214 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.436954866525023 }, { 1, 0.563045133474978 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationAdaBoostModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learner = new ClassificationAdaBoostLearner(10, 1, 3);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, 
                { "AptitudeTestScore", 24.0268096428771 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void ClassificationAdaBoostModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new ClassificationAdaBoostLearner(10, 1, 3);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 0.65083327864662022, 2.7087794356399844 };

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
