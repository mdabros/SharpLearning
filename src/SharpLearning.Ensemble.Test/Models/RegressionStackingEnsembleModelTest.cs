using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Common.Interfaces;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Ensemble.Test.Models
{
    /// <summary>
    /// Summary description for RegressionStackingEnsembleModelTest
    /// </summary>
    [TestClass]
    public class RegressionStackingEnsembleModelTest
    {
        [TestMethod]
        public void RegressionStackingEnsembleModel_Predict_single()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var learners = new IIndexedLearner<double>[]
            {
                new RegressionDecisionTreeLearner(2),
                new RegressionDecisionTreeLearner(5),
                new RegressionDecisionTreeLearner(7),
                new RegressionDecisionTreeLearner(9)
            };

            var learner = new RegressionStackingEnsembleLearner(learners,
                new RegressionDecisionTreeLearner(9),
                new RandomCrossValidation<double>(5, 23), false);

            var sut = learner.Learn(observations, targets);

            var rows = targets.Length;
            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.Row(i));
            }

            var metric = new MeanSquaredErrorRegressionMetric();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.26175213675213671, actual, 0.0000001);
        }

        [TestMethod]
        public void RegressionStackingEnsembleModel_Predict_Multiple()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var learners = new IIndexedLearner<double>[]
            {
                new RegressionDecisionTreeLearner(2),
                new RegressionDecisionTreeLearner(5),
                new RegressionDecisionTreeLearner(7),
                new RegressionDecisionTreeLearner(9)
            };

            var learner = new RegressionStackingEnsembleLearner(learners,
                new RegressionDecisionTreeLearner(9),
                new RandomCrossValidation<double>(5, 23), false);

            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var metric = new MeanSquaredErrorRegressionMetric();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.26175213675213671, actual, 0.0000001);
        }

        [TestMethod]
        public void RegressionStackingEnsembleModel_GetVariableImportance()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 },
                { "PreviousExperience_month", 1 } };

            var learners = new IIndexedLearner<double>[]
            {
                new RegressionDecisionTreeLearner(2),
                new RegressionDecisionTreeLearner(5),
                new RegressionDecisionTreeLearner(7),
                new RegressionDecisionTreeLearner(9)
            };

            var learner = new RegressionStackingEnsembleLearner(learners,
                new RegressionDecisionTreeLearner(9),
                new RandomCrossValidation<double>(5, 23), false);

            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "RegressionDecisionTreeModel_2", 100 }, { "RegressionDecisionTreeModel_1", 69.7214491857349 }, { "RegressionDecisionTreeModel_0", 33.8678328474247 }, { "RegressionDecisionTreeModel_3", 1.70068027210884 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void RegressionStackingEnsembleModel_GetRawVariableImportance()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var learners = new IIndexedLearner<double>[]
            {
                new RegressionDecisionTreeLearner(2),
                new RegressionDecisionTreeLearner(5),
                new RegressionDecisionTreeLearner(7),
                new RegressionDecisionTreeLearner(9)
            };

            var learner = new RegressionStackingEnsembleLearner(learners,
                new RegressionDecisionTreeLearner(9),
                new RandomCrossValidation<double>(5, 23), false);

            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 0.255311355311355, 0.525592463092463, 0.753846153846154, 0.0128205128205128 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        static void WriteRawImportances(double[] featureImportance)
        {
            var result = "new double[] {";
            foreach (var item in featureImportance)
            {
                result += item + ", ";
            }

            Trace.WriteLine(result);
        }

        static void WriteImportances(Dictionary<string, double> featureImportance)
        {
            var result = "new Dictionary<string, double> {";
            foreach (var item in featureImportance)
            {
                result += "{" + "\"" + item.Key + "\"" + ", " + item.Value + "}, ";
            }

            Trace.WriteLine(result);
        }
    }
}
