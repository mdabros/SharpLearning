using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Metrics.Test.Classification;

[TestClass]
public class RocAucClassificationProbabilityMetricTest
{
    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void RocAucClassificationMetric_Error_Not_Binary()
    {
        var targets = new double[] { 0, 1, 2 };
        var probabilities = new ProbabilityPrediction[0];

        var sut = new RocAucClassificationProbabilityMetric(1);
        var actual = sut.Error(targets, probabilities);
    }

    [TestMethod]
    public void RocAucClassificationMetric_Error_No_Error()
    {
        var targets = new double[] { 0, 1 };
        var probabilities = new ProbabilityPrediction[] { new(0, new Dictionary<double, double> { { 0, 0 }, { 1.0, 0.0 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 1 } }) };
        var sut = new RocAucClassificationProbabilityMetric(1);
        var actual = sut.Error(targets, probabilities);

        Assert.AreEqual(0.0, actual);
    }

    [TestMethod]
    public void RocAucClassificationMetric_Error()
    {
        var targets = new double[] { 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1 };
        var probabilities = new ProbabilityPrediction[] { new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.052380952 } }), new(0, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.020725389 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.993377483 } }), new(0, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.020725389 } }), new(0, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.020725389 } }), new(0, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.111111111 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.193377483 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.793377483 } }), new(0, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.020725389 } }), new(0, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.012345679 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.885860173 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.714285714 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.985860173 } }), new(0, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.020725389 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.985860173 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.993377483 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.993377483 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.954545455 } }), new(0, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.020725389 } }), new(0, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.020725389 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.985860173 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 0.985860173 } }) };

        var sut = new RocAucClassificationProbabilityMetric(1);
        var actual = sut.Error(targets, probabilities);
        Assert.AreEqual(0.0085470085470086277, actual, 0.00001);
    }

    [TestMethod]
    public void RocAucClassificationMetric_Error_Random()
    {
        var positives = Enumerable.Range(0, 800).Select(s => 1.0).ToList();
        var negatives = Enumerable.Range(0, 800).Select(s => 0.0).ToList();
        positives.AddRange(negatives);
        var targets = positives.ToArray();

        var random = new Random(42);
        var probabilities = targets
        .Select(s => s == 0 ? new ProbabilityPrediction(0.0, new Dictionary<double, double> { { 0.0, random.NextDouble() }, { 1.0, random.NextDouble() } }) :
                              new ProbabilityPrediction(1.0, new Dictionary<double, double> { { 0.0, random.NextDouble() }, { 1.0, random.NextDouble() } }))
        .ToArray();

        var sut = new RocAucClassificationProbabilityMetric(1);
        var actual = sut.Error(targets, probabilities);

        Assert.AreEqual(0.507653125, actual, 0.0001);
    }

    [TestMethod]
    public void RocAucClassificationMetric_Error_Always_Negative()
    {
        var positives = Enumerable.Range(0, 200).Select(s => 1.0).ToList();
        var negatives = Enumerable.Range(0, 800).Select(s => 0.0).ToList();
        positives.AddRange(negatives);
        var targets = positives.ToArray();

        var probabilities = targets
            .Select(s => s == 0 ? new ProbabilityPrediction(0.0, new Dictionary<double, double> { { 0.0, 0.0 }, { 1.0, 0.0 } }) :
                                  new ProbabilityPrediction(1.0, new Dictionary<double, double> { { 0.0, 0.0 }, { 1.0, 0.0 } }))
            .ToArray();

        var sut = new RocAucClassificationProbabilityMetric(1);
        var actual = sut.Error(targets, probabilities);

        Assert.AreEqual(0.5, actual, 0.0001);
    }

    [TestMethod]
    public void RocAucClassificationMetric_Error_Always_Positve()
    {
        var positives = Enumerable.Range(0, 800).Select(s => 1.0).ToList();
        var negatives = Enumerable.Range(0, 200).Select(s => 0.0).ToList();
        positives.AddRange(negatives);
        var targets = positives.ToArray();

        var probabilities = targets
            .Select(s => s == 0 ? new ProbabilityPrediction(0.0, new Dictionary<double, double> { { 0.0, 1.0 }, { 1.0, 1.0 } }) :
                                  new ProbabilityPrediction(1.0, new Dictionary<double, double> { { 0.0, 1.0 }, { 1.0, 1.0 } }))
            .ToArray();

        var sut = new RocAucClassificationProbabilityMetric(1);
        var actual = sut.Error(targets, probabilities);

        Assert.AreEqual(0.5, actual, 0.0001);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void RocAucClassificationMetric_Error_Only_Positve_Targets()
    {
        var positives = Enumerable.Range(0, 10).Select(s => 1.0).ToList();
        var targets = positives.ToArray();

        var probabilities = targets
            .Select(s => new ProbabilityPrediction(0.0, new Dictionary<double, double> { { 0.0, 1.0 }, { 1.0, 0.0 } }))
            .ToArray();

        var sut = new RocAucClassificationProbabilityMetric(1);
        sut.Error(targets, probabilities);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void RocAucClassificationMetric_Error_Only_Negative_Targets()
    {
        var positives = Enumerable.Range(0, 10).Select(s => 0.0).ToList();
        var targets = positives.ToArray();

        var probabilities = targets
            .Select(s => new ProbabilityPrediction(0.0, new Dictionary<double, double> { { 0.0, 1.0 }, { 1.0, 0.0 } }))
            .ToArray();

        var sut = new RocAucClassificationProbabilityMetric(1);
        sut.Error(targets, probabilities);
    }

    [TestMethod]
    public void RocAucClassificationMetric_ErrorString()
    {
        var targets = new double[] { 0, 1 };
        var probabilities = new ProbabilityPrediction[] { new(0, new Dictionary<double, double> { { 0, 0 }, { 1.0, 0.0 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 1 } }) };
        var sut = new RocAucClassificationProbabilityMetric(1);
        var actual = sut.ErrorString(targets, probabilities);

        var expected = ";0;1;0;1\r\n0;1.000;0.000;100.000;0.000\r\n1;0.000;1.000;0.000;100.000\r\nError: 0.000\r\n";

        Assert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void RocAucClassificationMetric_ErrorString_TargetStringMapping()
    {
        var targets = new double[] { 0, 1 };
        var probabilities = new ProbabilityPrediction[] { new(0, new Dictionary<double, double> { { 0, 0 }, { 1.0, 0.0 } }), new(1, new Dictionary<double, double> { { 0, 0.0 }, { 1.0, 1 } }) };
        var sut = new RocAucClassificationProbabilityMetric(1);
        var targetStringMapping = new Dictionary<double, string>
        {
            { 0, "Negative" },
            { 1, "Positive" },
        };

        var actual = sut.ErrorString(targets, probabilities, targetStringMapping);
        var expected = ";Negative;Positive;Negative;Positive\r\nNegative;1.000;0.000;100.000;0.000\r\nPositive;0.000;1.000;0.000;100.000\r\nError: 0.000\r\n";

        Assert.AreEqual(expected, actual);
    }
}
