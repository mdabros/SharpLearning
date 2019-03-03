using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.Containers.Extensions;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.AdaBoost.Test.Learners
{
    [TestClass]
    public class ClassificationAdaBoostLearnerTest
    {
        [TestMethod]
        public void ClassificationAdaBoostLearner_Learn_AptitudeData()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var sut = new ClassificationAdaBoostLearner(10);
            
            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038461538461538464, actual);
        }

        [TestMethod]
        public void ClassificationAdaBoostLearner_Learn_Glass()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new ClassificationAdaBoostLearner(10, 1, 5);

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.0, actual);
        }

        [TestMethod]
        public void ClassificationAdaBoostLearner_Learn_AptitudeData_SequenceContainNoItemIssue_Solved()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();
            var indices = new int[] { 22, 6, 23, 12 };

            var sut = new ClassificationAdaBoostLearner(10);
            
            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);
            var indexedPredictions = predictions.GetIndices(indices);
            var indexedTargets = targets.GetIndices(indices);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(indexedTargets, indexedPredictions);

            Assert.AreEqual(0.0, actual);
        }

        [TestMethod]
        public void ClassificationAdaBoostLearner_Learn_Glass_Indexed()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new ClassificationAdaBoostLearner(10, 1, 5);

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);
            var indexedPredictions = predictions.GetIndices(indices);
            var indexedTargets = targets.GetIndices(indices);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(indexedTargets, indexedPredictions);

            Assert.AreEqual(0.0, actual);
        }
    }
}
