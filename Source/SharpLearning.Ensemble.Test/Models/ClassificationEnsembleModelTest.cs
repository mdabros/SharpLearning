using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Ensemble.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpLearning.Ensemble.Test.Models
{
    [TestClass]
    public class ClassificationEnsembleModelTest
    {
        [TestMethod]
        public void ClassificationEnsembleModel_Predict_single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationEnsembleLearner(learners, new MeanProbabilityClassificationEnsembleStrategy());
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.Row(i));
            }

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.076923076923076927, actual, 0.0000001);
        }

        [TestMethod]
        public void ClassificationEnsembleModel_Predict_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationEnsembleLearner(learners, new MeanProbabilityClassificationEnsembleStrategy());
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.076923076923076927, actual, 0.0000001);
        }

        [TestMethod]
        public void ClassificationEnsembleModel_PredictProbability_single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationEnsembleLearner(learners, new MeanProbabilityClassificationEnsembleStrategy());
            var sut = learner.Learn(observations, targets);

            var predictions = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.PredictProbability(observations.Row(i));
            }

            var metric = new LogLossClassificationProbabilityMetric();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.32562112824941963, actual, 0.0000001);
        }

        [TestMethod]
        public void ClassificationEnsembleModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationEnsembleLearner(learners, new MeanProbabilityClassificationEnsembleStrategy());
            var sut = learner.Learn(observations, targets);

            var predictions = sut.PredictProbability(observations);

            var metric = new LogLossClassificationProbabilityMetric();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.32562112824941963, actual, 0.0000001);
        }

        [TestMethod]
        public void ClassificationEnsembleModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationEnsembleLearner(learners, new MeanProbabilityClassificationEnsembleStrategy());
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, 
                { "AptitudeTestScore", 15.6771501925546 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void ClassificationEnsembleModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationEnsembleLearner(learners, new MeanProbabilityClassificationEnsembleStrategy());
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 100.0, 15.6771501925546 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }
    }
}
