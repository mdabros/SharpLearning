using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.BiasVarianceAnalysis;
using System.Collections.Generic;

namespace SharpLearning.CrossValidation.Test.BiasVarianceAnalysis
{
    [TestClass]
    public class RandomBiasVarianceLearningCurvesCalculatorTest
    {
        [TestMethod]
        public void RandomBiasVarianceLearningCurvesCalculator_Calculate()
        {
            var sut = new RandomBiasVarianceLearningCurvesCalculator<double>(new CrossValidationTestMetric(), 
                new double[] { 0.2, 0.8 }, 0.8, 42);

            var observations = new F64Matrix(10, 10);
            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var indices = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            var actual = sut.Calculate(() => new CrossValidationTestLearner(indices),
                observations, targets);

            var expected = new List<BiasVarianceLearningCurvePoint>() { new BiasVarianceLearningCurvePoint(1, 81, 8), 
                new BiasVarianceLearningCurvePoint(6, 24.166666666666664, 14.5)};

            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
