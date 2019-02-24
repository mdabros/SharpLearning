using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Ensemble.EnsembleSelectors;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Ensemble.Test.EnsembleSelectors
{
    [TestClass]
    public class BackwardEliminationClassificationEnsembleSelectionTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void BackwardEliminationClassificationEnsembleSelection_Constructor_Metric_Null()
        {
            var sut = new BackwardEliminationClassificationEnsembleSelection(null, new MeanProbabilityClassificationEnsembleStrategy(), 5);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void BackwardEliminationClassificationEnsembleSelection_Constructor_EnsembleStratey_Null()
        {
            var sut = new BackwardEliminationClassificationEnsembleSelection(new LogLossClassificationProbabilityMetric(), null, 5);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BackwardEliminationClassificationEnsembleSelection_Constructor_Number_Of_Models_Too_Low()
        {
            var sut = new BackwardEliminationClassificationEnsembleSelection(new LogLossClassificationProbabilityMetric(), 
                new MeanProbabilityClassificationEnsembleStrategy(), 0);
        }

        [TestMethod]
        public void BackwardEliminationClassificationEnsembleSelection_Select()
        {
            var sut = new BackwardEliminationClassificationEnsembleSelection(new LogLossClassificationProbabilityMetric(),
                new MeanProbabilityClassificationEnsembleStrategy(), 3);

            var random = new Random(42);

            var observations = CreateModelPredictions(random);
            var targets = Enumerable.Range(0, 10).Select(v => (double)random.Next(1)).ToArray();

            var actual = sut.Select(observations, targets);
            var expected = new int[] { 1, 7 };

            CollectionAssert.AreEqual(expected, actual);
        }

        static ProbabilityPrediction[][] CreateModelPredictions(Random random)
        {
            return new ProbabilityPrediction[10][]
            .Select(t => Enumerable.Range(0, 10)
            .Select(p => new ProbabilityPrediction(random.Next(1), new Dictionary<double, double> { { 0, random.NextDouble() }, { 1, random.NextDouble() } }))
            .ToArray()).ToArray();
        }
    }
}
