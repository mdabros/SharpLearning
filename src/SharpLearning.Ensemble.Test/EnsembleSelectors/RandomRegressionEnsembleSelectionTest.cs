using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Ensemble.EnsembleSelectors;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Ensemble.Test.EnsembleSelectors
{
    [TestClass]
    public class RandomRegressionEnsembleSelectionTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RandomRegressionEnsembleSelection_Constructor_Metric_Null()
        {
            var sut = new RandomRegressionEnsembleSelection(null, new MeanRegressionEnsembleStrategy(), 5, 1, true);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RandomRegressionEnsembleSelection_Constructor_EnsembleStratey_Null()
        {
            var sut = new RandomRegressionEnsembleSelection(new MeanSquaredErrorRegressionMetric(), null, 5, 1, true);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RandomRegressionEnsembleSelection_Constructor_Number_Of_Models_Too_Low()
        {
            var sut = new RandomRegressionEnsembleSelection(new MeanSquaredErrorRegressionMetric(), 
                new MeanRegressionEnsembleStrategy(), 0, 1, true);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RandomRegressionEnsembleSelection_Constructor_Number_Of_Models_From_Start_Too_Low()
        {
            var sut = new RandomRegressionEnsembleSelection(new MeanSquaredErrorRegressionMetric(),
                new MeanRegressionEnsembleStrategy(), 5, 0, true);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RandomRegressionEnsembleSelection_Constructor_Number_Of_Availible_Models_Lower_Than_Number_Of_Models_To_Select()
        {
            var sut = new RandomRegressionEnsembleSelection(new MeanSquaredErrorRegressionMetric(),
                new MeanRegressionEnsembleStrategy(), 5, 1, true);

            var observations = new F64Matrix(10, 3);
            var targets = new double[10];

            sut.Select(observations, targets);
        }

        [TestMethod]
        public void RandomRegressionEnsembleSelection_Select()
        {
            var sut = new RandomRegressionEnsembleSelection(new MeanSquaredErrorRegressionMetric(),
                new MeanRegressionEnsembleStrategy(), 3, 10, true);

            var random = new Random(42);

            var observations = new F64Matrix(10, 10);
            observations.Map(() => random.Next());
            var targets = Enumerable.Range(0, 10).Select(v => random.NextDouble()).ToArray();

            var actual = sut.Select(observations, targets);
            var expected = new int[3] { 2, 0, 2 };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void RandomRegressionEnsembleSelection_Select_No_Replacements()
        {
            var sut = new RandomRegressionEnsembleSelection(new MeanSquaredErrorRegressionMetric(),
                new MeanRegressionEnsembleStrategy(), 3, 10, false);

            var random = new Random(42);

            var observations = new F64Matrix(10, 10);
            observations.Map(() => random.Next());
            var targets = Enumerable.Range(0, 10).Select(v => random.NextDouble()).ToArray();

            var actual = sut.Select(observations, targets);
            var expected = new int[3] { 3, 8, 2 };

            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
