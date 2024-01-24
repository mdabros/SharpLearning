using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.EnsembleSelectors;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Ensemble.Test.Learners
{
    [TestClass]
    public class ClassificationModelSelectingEnsembleLearnerTest
    {
        [TestMethod]
        public void ClassificationModelSelectingEnsembleLearner_Learn()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9),
                new ClassificationDecisionTreeLearner(11),
                new ClassificationDecisionTreeLearner(21),
                new ClassificationDecisionTreeLearner(23),
                new ClassificationDecisionTreeLearner(1),
                new ClassificationDecisionTreeLearner(14),
                new ClassificationDecisionTreeLearner(17),
                new ClassificationDecisionTreeLearner(19),
                new ClassificationDecisionTreeLearner(33)
            };

            var metric = new LogLossClassificationProbabilityMetric();
            var ensembleStrategy = new MeanProbabilityClassificationEnsembleStrategy();
            var ensembleSelection = new ForwardSearchClassificationEnsembleSelection(
                metric, ensembleStrategy, 5, 1, true);

            var sut = new ClassificationModelSelectingEnsembleLearner(learners,
                new RandomCrossValidation<ProbabilityPrediction>(5, 23),
                ensembleStrategy, ensembleSelection);

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var model = sut.Learn(observations, targets);
            var predictions = model.PredictProbability(observations);

            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.55183985816428427, actual, 0.0001);
        }

        [TestMethod]
        public void ClassificationModelSelectingEnsembleLearner_Learn_Without_Replacement()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9),
                new ClassificationDecisionTreeLearner(11),
                new ClassificationDecisionTreeLearner(21),
                new ClassificationDecisionTreeLearner(23),
                new ClassificationDecisionTreeLearner(1),
                new ClassificationDecisionTreeLearner(14),
                new ClassificationDecisionTreeLearner(17),
                new ClassificationDecisionTreeLearner(19),
                new ClassificationDecisionTreeLearner(33)
            };

            var metric = new LogLossClassificationProbabilityMetric();
            var ensembleStrategy = new MeanProbabilityClassificationEnsembleStrategy();
            var ensembleSelection = new ForwardSearchClassificationEnsembleSelection(
                metric, ensembleStrategy, 5, 1, false);

            var sut = new ClassificationModelSelectingEnsembleLearner(learners,
                new RandomCrossValidation<ProbabilityPrediction>(5, 23),
                ensembleStrategy, ensembleSelection);

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var model = sut.Learn(observations, targets);
            var predictions = model.PredictProbability(observations);

            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.52718235404723557, actual, 0.0001);
        }

        [TestMethod]
        public void ClassificationModelSelectingEnsembleLearner_Learn_Start_With_3_Models()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9),
                new ClassificationDecisionTreeLearner(11),
                new ClassificationDecisionTreeLearner(21),
                new ClassificationDecisionTreeLearner(23),
                new ClassificationDecisionTreeLearner(1),
                new ClassificationDecisionTreeLearner(14),
                new ClassificationDecisionTreeLearner(17),
                new ClassificationDecisionTreeLearner(19),
                new ClassificationDecisionTreeLearner(33)
            };

            var metric = new LogLossClassificationProbabilityMetric();
            var ensembleStrategy = new MeanProbabilityClassificationEnsembleStrategy();
            var ensembleSelection = new ForwardSearchClassificationEnsembleSelection(metric,
                ensembleStrategy, 5, 3, true);

            var sut = new ClassificationModelSelectingEnsembleLearner(learners,
                new RandomCrossValidation<ProbabilityPrediction>(5, 23),
                ensembleStrategy, ensembleSelection);

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var model = sut.Learn(observations, targets);
            var predictions = model.PredictProbability(observations);

            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.55183985816428427, actual, 0.0001);
        }

        [TestMethod]
        public void ClassificationModelSelectingEnsembleLearner_Learn_Indexed()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9),
                new ClassificationDecisionTreeLearner(11),
                new ClassificationDecisionTreeLearner(21),
                new ClassificationDecisionTreeLearner(23),
                new ClassificationDecisionTreeLearner(1),
                new ClassificationDecisionTreeLearner(14),
                new ClassificationDecisionTreeLearner(17),
                new ClassificationDecisionTreeLearner(19),
                new ClassificationDecisionTreeLearner(33)
            };

            var metric = new LogLossClassificationProbabilityMetric();
            var ensembleStrategy = new MeanProbabilityClassificationEnsembleStrategy();
            var ensembleSelection = new ForwardSearchClassificationEnsembleSelection(metric,
                ensembleStrategy, 5, 1, true);

            var sut = new ClassificationModelSelectingEnsembleLearner(learners,
                new RandomCrossValidation<ProbabilityPrediction>(5, 23),
                ensembleStrategy, ensembleSelection);

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var indices = Enumerable.Range(0, 25).ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.PredictProbability(observations);

            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(2.3682546920482164, actual, 0.0001);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ClassificationModelSelectingEnsembleLearner_Constructor_Learners_Null()
        {
            var metric = new LogLossClassificationProbabilityMetric();
            var ensembleStrategy = new MeanProbabilityClassificationEnsembleStrategy();
            var ensembleSelection = new ForwardSearchClassificationEnsembleSelection(metric,
                ensembleStrategy, 5, 1, true);
            var crossValidation = new RandomCrossValidation<ProbabilityPrediction>(5);

            var sut = new ClassificationModelSelectingEnsembleLearner(null, crossValidation,
                ensembleStrategy, ensembleSelection);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ClassificationModelSelectingEnsembleLearner_Constructor_CrossValidation_Null()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[4];
            var metric = new LogLossClassificationProbabilityMetric();
            var ensembleStrategy = new MeanProbabilityClassificationEnsembleStrategy();
            var ensembleSelection = new ForwardSearchClassificationEnsembleSelection(metric,
                ensembleStrategy, 5, 1, true);

            var sut = new ClassificationModelSelectingEnsembleLearner(learners, null,
                ensembleStrategy, ensembleSelection);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ClassificationModelSelectingEnsembleLearner_Constructor_EnsembleSelection_Null()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[4];
            var metric = new LogLossClassificationProbabilityMetric();
            var ensembleStrategy = new MeanProbabilityClassificationEnsembleStrategy();
            var crossValidation = new RandomCrossValidation<ProbabilityPrediction>(5);

            var sut = new ClassificationModelSelectingEnsembleLearner(learners, crossValidation,
                ensembleStrategy, null);
        }
    }
}
