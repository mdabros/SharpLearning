using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Ensemble.Test.Learners
{
    [TestClass]
    public class ClassificationStackingEnsembleLearnerTest
    {
        [TestMethod]
        public void ClassificationStackingEnsembleLearner_Learn()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var sut = new ClassificationStackingEnsembleLearner(learners, 
                new ClassificationDecisionTreeLearner(9),
                new RandomCrossValidation<ProbabilityPrediction>(5, 23), false);

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.63551401869158874, actual, 0.0001);
        }

        [TestMethod]
        public void ClassificationStackingEnsembleLearner_CreateMetaFeatures_Then_Learn()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var sut = new ClassificationStackingEnsembleLearner(learners, 
                new ClassificationDecisionTreeLearner(9),
                new RandomCrossValidation<ProbabilityPrediction>(5, 23), false);

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var metaObservations = sut.LearnMetaFeatures(observations, targets);
            var model = sut.LearnStackingModel(observations, metaObservations, targets);

            var predictions = model.Predict(observations);

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.63551401869158874, actual, 0.0001);
        }

        [TestMethod]
        public void ClassificationStackingEnsembleLearner_Learn_Include_Original_Features()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var sut = new ClassificationStackingEnsembleLearner(learners, 
                new ClassificationDecisionTreeLearner(9),
                new RandomCrossValidation<ProbabilityPrediction>(5, 23), true);

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.26168224299065418, actual, 0.0001);
        }

        [TestMethod]
        public void ClassificationStackingEnsembleLearner_Learn_Indexed()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9)
            };

            var sut = new ClassificationStackingEnsembleLearner(learners, 
                new ClassificationDecisionTreeLearner(9),
                new RandomCrossValidation<ProbabilityPrediction>(5, 23), false);

            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var indices = Enumerable.Range(0, 25).ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.67289719626168221, actual, 0.0001);
        }
    }
}
