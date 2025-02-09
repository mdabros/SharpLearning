using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.LearningCurves;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.CrossValidation.Test.LearningCurves;

[TestClass]
public class StratifiedLearningCurvesCalculatorTest
{
    [TestMethod]
    public void StratifiedLearningCurvesCalculator_Calculate()
    {
        var sut = new StratifiedLearningCurvesCalculator<double>(
            new TotalErrorClassificationMetric<double>(),
            new double[] { 0.2, 0.8 },
            0.8, 5, 42);

        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var actual = sut.Calculate(new ClassificationDecisionTreeLearner(),
            observations, targets);

        var expected = new List<LearningCurvePoint>()
        {
            new(4, 0, 0.39999999999999997),
            new(16, 0.0625, 0.33333333333333331)
        };

        CollectionAssert.AreEqual(expected, actual);
    }
}
