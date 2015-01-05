using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.BiasVarianceAnalysis;
using System.Collections.Generic;

namespace SharpLearning.CrossValidation.Test.BiasVarianceAnalysis
{
    [TestClass]
    public class NoShuffleBiasVarianceLearningCurvesCalculatorTest
    {
        [TestMethod]
        public void NoShuffleBiasVarianceLearningCurvesCalculator_Calculate()
        {
            var sut = new NoShuffleBiasVarianceLearningCurvesCalculator<double>(new CrossValidationTestMetric(), 
                new double[] { 0.2, 0.8 }, 0.8 );

            var observations = new F64Matrix(10, 10);
            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var indices = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            var actual = sut.Calculate(() => new CrossValidationTestLearner(indices),
                observations, targets);

            var expected = new List<BiasVarianceLearningCurvePoint>() { new BiasVarianceLearningCurvePoint(1, 1, 36), 
                new BiasVarianceLearningCurvePoint(6, 25.333333333333332, 0)};
            
            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
