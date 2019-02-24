using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Ensemble.Test.Learners
{
    [TestClass]
    public class ClassificationBackwardEliminationModelSelectingEnsembleLearnerTest
    {
        [TestMethod]
        public void ClassificationBackwardEliminationModelSelectingEnsembleLearner_Learn()
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

            var sut = new ClassificationBackwardEliminationModelSelectingEnsembleLearner(learners, 5);

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var model = sut.Learn(observations, targets);
            var predictions = model.PredictProbability(observations);

            var metric = new LogLossClassificationProbabilityMetric();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.52351727716455632, actual, 0.0001);
        }

        [TestMethod]
        public void ClassificationBackwardEliminationModelSelectingEnsembleLearner_CreateMetaFeatures_Then_Learn()
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

            var sut = new ClassificationBackwardEliminationModelSelectingEnsembleLearner(learners, 5);

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var metaObservations = sut.LearnMetaFeatures(observations, targets);
            var model = sut.SelectModels(observations, metaObservations, targets);

            var predictions = model.PredictProbability(observations);

            var metric = new LogLossClassificationProbabilityMetric();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.52351727716455632, actual, 0.0001);
        }

        [TestMethod]
        public void ClassificationBackwardEliminationModelSelectingEnsembleLearner_Learn_Indexed()
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

            var sut = new ClassificationBackwardEliminationModelSelectingEnsembleLearner(learners, 5,
                new RandomCrossValidation<ProbabilityPrediction>(5, 23), ensembleStrategy, metric);

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var indices = Enumerable.Range(0, 25).ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.PredictProbability(observations);

            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(2.3682546920482164, actual, 0.0001);
        }
    }
}
