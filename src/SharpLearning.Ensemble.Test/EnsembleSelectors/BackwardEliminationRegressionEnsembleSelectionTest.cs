using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Ensemble.EnsembleSelectors;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Ensemble.Test.EnsembleSelectors;

[TestClass]
public class BackwardEliminationRegressionEnsembleSelectionTest
{
    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void BackwardEliminationRegressionEnsembleSelection_Constructor_Metric_Null()
    {
        var sut = new BackwardEliminationRegressionEnsembleSelection(null,
            new MeanRegressionEnsembleStrategy(), 5);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void BackwardEliminationRegressionEnsembleSelection_Constructor_EnsembleStratey_Null()
    {
        var sut = new BackwardEliminationRegressionEnsembleSelection(
            new MeanSquaredErrorRegressionMetric(), null, 5);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void BackwardEliminationRegressionEnsembleSelection_Constructor_Number_Of_Models_Too_Low()
    {
        var sut = new BackwardEliminationRegressionEnsembleSelection(
            new MeanSquaredErrorRegressionMetric(),
            new MeanRegressionEnsembleStrategy(), 0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void BackwardEliminationRegressionEnsembleSelection_Constructor_Number_Of_Availible_Models_Lower_Than_Number_Of_Models_To_Select()
    {
        var sut = new BackwardEliminationRegressionEnsembleSelection(
            new MeanSquaredErrorRegressionMetric(),
            new MeanRegressionEnsembleStrategy(), 5);

        var observations = new F64Matrix(10, 3);
        var targets = new double[10];

        sut.Select(observations, targets);
    }

    [TestMethod]
    public void BackwardEliminationRegressionEnsembleSelection_Select()
    {
        var sut = new BackwardEliminationRegressionEnsembleSelection(
            new MeanSquaredErrorRegressionMetric(),
            new MeanRegressionEnsembleStrategy(), 3);

        var random = new Random(42);

        var observations = new F64Matrix(10, 10);
        observations.Map(() => random.Next());
        var targets = Enumerable.Range(0, 10).Select(v => random.NextDouble()).ToArray();

        var actual = sut.Select(observations, targets);
        var expected = new int[] { 2 };

        CollectionAssert.AreEqual(expected, actual);
    }
}
