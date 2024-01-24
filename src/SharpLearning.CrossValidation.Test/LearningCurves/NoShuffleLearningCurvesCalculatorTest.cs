using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.LearningCurves;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.CrossValidation.Test.LearningCurves
{
    [TestClass]
    public class NoShuffleLearningCurvesCalculatorTest
    {
        [TestMethod]
        public void NoShuffleLearningCurvesCalculator_Calculate()
        {
            var sut = new NoShuffleLearningCurvesCalculator<double>(
                new MeanSquaredErrorRegressionMetric(),
                new double[] { 0.2, 0.8 },
                0.8);

            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var actual = sut.Calculate(new RegressionDecisionTreeLearner(),
                observations, targets);

            var expected = new List<LearningCurvePoint>()
            {
                new(32, 0, 0.12874833873980004),
                new(128, 0.0, 0.067720786718774989)
            };

            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
