using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Ensemble.Strategies;

namespace SharpLearning.Ensemble.Test.Strategies;

[TestClass]
public class GeometricMeanProbabilityClassificationEnsembleStrategyTest
{
    [TestMethod]
    public void GeometricMeanProbabilityClassificationEnsembleStrategy_Combine()
    {
        var values = new ProbabilityPrediction[]
        {
            new(1.0, new Dictionary<double,double> { {0.0, 0.3}, {1.0, 0.88} }),
            new(0.0, new Dictionary<double,double> { {0.0, 0.66}, {1.0, 0.33} }),
            new(1.0, new Dictionary<double,double> { {0.0, 0.01}, {1.0, 0.99} }),
        };

        var sut = new GeometricMeanProbabilityClassificationEnsembleStrategy();
        var actual = sut.Combine(values);

        var expected = new ProbabilityPrediction(1.0, new Dictionary<double, double> { { 0.0, 0.159846490962181 }, { 1.0, 0.840153509037819 } });
        Assert.AreEqual(expected, actual);
    }
}
