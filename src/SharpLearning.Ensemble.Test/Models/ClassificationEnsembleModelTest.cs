using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Ensemble.Test.Models
{
    [TestClass]
    public class ClassificationEnsembleModelTest
    {
        [TestMethod]
        public void ClassificationEnsembleModel_Predict_single()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationEnsembleLearner(learners, 
                new MeanProbabilityClassificationEnsembleStrategy());

            var sut = learner.Learn(observations, targets);

            var rows = targets.Length;
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
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationEnsembleLearner(learners, 
                new MeanProbabilityClassificationEnsembleStrategy());

            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.076923076923076927, actual, 0.0000001);
        }

        [TestMethod]
        public void ClassificationEnsembleModel_PredictProbability_single()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationEnsembleLearner(learners, 
                new MeanProbabilityClassificationEnsembleStrategy());

            var sut = learner.Learn(observations, targets);

            var rows = targets.Length;
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
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationEnsembleLearner(learners, 
                new MeanProbabilityClassificationEnsembleStrategy());

            var sut = learner.Learn(observations, targets);

            var predictions = sut.PredictProbability(observations);

            var metric = new LogLossClassificationProbabilityMetric();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.32562112824941963, actual, 0.0000001);
        }

        [TestMethod]
        public void ClassificationEnsembleModel_GetVariableImportance()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationEnsembleLearner(learners, 
                new MeanProbabilityClassificationEnsembleStrategy());

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
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var learner = new ClassificationEnsembleLearner(learners, 
                new MeanProbabilityClassificationEnsembleStrategy());

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
