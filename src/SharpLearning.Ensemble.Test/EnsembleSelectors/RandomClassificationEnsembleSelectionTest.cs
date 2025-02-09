using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Ensemble.EnsembleSelectors;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Ensemble.Test.EnsembleSelectors;

[TestClass]
public class RandomClassificationEnsembleSelectionTest
{
    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void RandomClassificationEnsembleSelection_Constructor_Metric_Null()
    {
        var sut = new RandomClassificationEnsembleSelection(null,
            new MeanProbabilityClassificationEnsembleStrategy(), 5, 1, true);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void RandomClassificationEnsembleSelection_Constructor_EnsembleStratey_Null()
    {
        var sut = new RandomClassificationEnsembleSelection(
            new LogLossClassificationProbabilityMetric(), null, 5, 1, true);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void RandomClassificationEnsembleSelection_Constructor_Number_Of_Models_Too_Low()
    {
        var sut = new RandomClassificationEnsembleSelection(
            new LogLossClassificationProbabilityMetric(),
            new MeanProbabilityClassificationEnsembleStrategy(), 0, 1, true);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void RandomClassificationEnsembleSelection_Constructor_Number_Of_Availible_Models_Lower_Than_Number_Of_Models_To_Select()
    {
        var sut = new RandomClassificationEnsembleSelection(
            new LogLossClassificationProbabilityMetric(),
            new MeanProbabilityClassificationEnsembleStrategy(), 5, 1, true);

        var observations = new ProbabilityPrediction[3][];
        observations.Select(t => new ProbabilityPrediction[10]).ToArray();
        var targets = new double[10];

        sut.Select(observations, targets);
    }

    [TestMethod]
    public void RandomClassificationEnsembleSelection_Select()
    {
        var sut = new RandomClassificationEnsembleSelection(
            new LogLossClassificationProbabilityMetric(),
            new MeanProbabilityClassificationEnsembleStrategy(), 3, 10, true);

        var random = new Random(42);

        var observations = CreateModelPredictions(random);
        var targets = Enumerable.Range(0, 10).Select(v => (double)random.Next(1)).ToArray();

        var actual = sut.Select(observations, targets);
        var expected = new int[3] { 1, 0, 1 };

        CollectionAssert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void RandomClassificationEnsembleSelection_Select_No_Replacements()
    {
        var sut = new RandomClassificationEnsembleSelection(
            new LogLossClassificationProbabilityMetric(),
            new MeanProbabilityClassificationEnsembleStrategy(), 3, 10, false);

        var random = new Random(42);

        var observations = CreateModelPredictions(random);
        var targets = Enumerable.Range(0, 10).Select(v => (double)random.Next(1)).ToArray();

        var actual = sut.Select(observations, targets);
        var expected = new int[] { 1, 0, 2 };

        CollectionAssert.AreEqual(expected, actual);
    }

    static ProbabilityPrediction[][] CreateModelPredictions(Random random)
    {
        return new ProbabilityPrediction[3][]
        .Select(t => Enumerable.Range(0, 10)
        .Select(p => new ProbabilityPrediction(random.Next(1), new Dictionary<double, double> { { 0, random.NextDouble() }, { 1, random.NextDouble() } }))
        .ToArray()).ToArray();
    }
}
