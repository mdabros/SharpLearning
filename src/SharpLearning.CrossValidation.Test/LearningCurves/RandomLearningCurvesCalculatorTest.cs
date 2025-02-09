using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.LearningCurves;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.CrossValidation.Test.LearningCurves;

[TestClass]
public class RandomLearningCurvesCalculatorTest
{
    [TestMethod]
    public void RandomLearningCurvesCalculator_Calculate()
    {
        var sut = new RandomShuffleLearningCurvesCalculator<double>(
            new MeanSquaredErrorRegressionMetric(),
            [0.2, 0.8],
            0.8, 42, 5);

        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var actual = sut.Calculate(new RegressionDecisionTreeLearner(),
            observations, targets);

        var expected = new List<LearningCurvePoint>()
        {
            new(32, 0, 0.141565953928265),
            new(128, 0.0, 0.068970597423950036)
        };

        CollectionAssert.AreEqual(expected, actual);
    }
}
