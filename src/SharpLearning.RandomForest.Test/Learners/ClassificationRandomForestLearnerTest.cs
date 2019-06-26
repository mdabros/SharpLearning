using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.Metrics.Classification;
using SharpLearning.RandomForest.Learners;

namespace SharpLearning.RandomForest.Test.Learners
{
    [TestClass]
    public class ClassificationRandomForestLearnerTest
    {
        readonly double m_delta = 0.0000001;

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Aptitude_Trees_1()
        {
            var error = ClassificationRandomLearner_Learn_Aptitude(1);
            Assert.AreEqual(0.42307692307692307, error, m_delta);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Aptitude_Trees_5()
        {
            var error = ClassificationRandomLearner_Learn_Aptitude(5);
            Assert.AreEqual(0.38461538461538464, error, m_delta);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Aptitude_Trees_100()
        {
            var error = ClassificationRandomLearner_Learn_Aptitude(100);
            Assert.AreEqual(0.23076923076923078, error, m_delta);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Aptitude_Trees_100_SubSample()
        {
            var error = ClassificationRandomLearner_Learn_Aptitude(100, 0.5);
            Assert.AreEqual(0.38461538461538464, error, m_delta);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Glass_1()
        {
            var error = ClassificationRandomForestLearner_Learn_Glass(1);
            Assert.AreEqual(0.20093457943925233, error, m_delta);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Glass_5()
        {
            var error = ClassificationRandomForestLearner_Learn_Glass(5);
            Assert.AreEqual(0.0560747663551402, error, m_delta);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Glass_100()
        {
            var error = ClassificationRandomForestLearner_Learn_Glass(100);
            Assert.AreEqual(0.018691588785046728, error, m_delta);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Glass_100_SubSample()
        {
            var error = ClassificationRandomForestLearner_Learn_Glass(100, 0.5);
            Assert.AreEqual(0.056074766355140186, error, m_delta);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Glass_100_Indices()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new ClassificationRandomForestLearner(100, 1, 100, 1, 0.0001, 1.0, 42, false);
            
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var model = sut.Learn(observations, targets, indices);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.098130841121495324, error, m_delta);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Glass_100_Trees_Parallel()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new ClassificationRandomForestLearner(100, 1, 100, 1, 0.0001, 1.0, 42, true);

            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.018691588785046728, error, m_delta);
        }

        double ClassificationRandomForestLearner_Learn_Glass(int trees, double subSampleRatio = 1.0)
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new ClassificationRandomForestLearner(trees, 1, 100, 1, 0.0001, 
                subSampleRatio, 42, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        double ClassificationRandomLearner_Learn_Aptitude(int trees, double subSampleRatio = 1.0)
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var sut = new ClassificationRandomForestLearner(trees, 5, 100, 1, 0.0001, 
                subSampleRatio, 42, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);
            return error;
        }
    }
}
