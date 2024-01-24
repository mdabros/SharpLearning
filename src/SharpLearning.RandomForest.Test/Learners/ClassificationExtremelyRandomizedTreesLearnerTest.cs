using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.Metrics.Classification;
using SharpLearning.RandomForest.Learners;

namespace SharpLearning.RandomForest.Test.Learners
{
    [TestClass]
    public class ClassificationExtremelyRandomizedTreesLearnerTest
    {
        readonly double m_delta = 0.0000001;

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude_Trees_1()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude(1);
            Assert.AreEqual(0.19230769230769232, error, m_delta);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude_Trees_5()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude(5);
            Assert.AreEqual(0.115384615384615, error, m_delta);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude_Trees_100()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude(100);
            Assert.AreEqual(0.153846153846154, error, m_delta);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude_Trees_100_SubSample()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude(100, 0.5);
            Assert.AreEqual(0.15384615384615386, error, m_delta);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Glass_1()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Glass(1);
            Assert.AreEqual(0.228971962616822, error, m_delta);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Glass_5()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Glass(5);
            Assert.AreEqual(0.0747663551401869, error, m_delta);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Glass_100()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Glass(100);
            Assert.AreEqual(0.0560747663551402, error, m_delta);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Glass_100_SubSample()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Glass(100, 0.5);
            Assert.AreEqual(0.0794392523364486, error, m_delta);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Glass_100_Indices()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new ClassificationExtremelyRandomizedTreesLearner(100, 1, 100, 1, 0.0001, 1.0, 42, false);

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var model = sut.Learn(observations, targets, indices);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.11214953271028, error, m_delta);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Glass_100_Trees_Parallel()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new ClassificationExtremelyRandomizedTreesLearner(100, 1, 100, 1, 0.0001, 1.0, 42, true);

            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.0560747663551402, error, m_delta);
        }

        static double ClassificationExtremelyRandomizedTreesLearner_Learn_Glass(int trees,
            double subSampleRatio = 1.0)
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new ClassificationExtremelyRandomizedTreesLearner(trees, 1, 100, 1, 0.0001,
                subSampleRatio, 42, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        static double ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude(int trees,
            double subSampleRatio = 1.0)
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var sut = new ClassificationExtremelyRandomizedTreesLearner(trees, 1, 100, 1, 0.0001,
                subSampleRatio, 42, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);
            return error;
        }
    }
}
