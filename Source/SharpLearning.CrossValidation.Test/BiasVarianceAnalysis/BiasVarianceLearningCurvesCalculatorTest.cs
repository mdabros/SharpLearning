using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.BiasVarianceAnalysis;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.CrossValidation.Test.BiasVarianceAnalysis
{
    [TestClass]
    public class BiasVarianceLearningCurvesCalculatorTest
    {
        [TestMethod]
        public void BiasVarianceLearningCurvesCalculator_Calculate()
        {
            var sut = new BiasVarianceLearningCurvesCalculator(
                new CrossValidationTestMetric(), new double[] { 0.2, 0.8 }, 0.8, 42);

            var observations = new F64Matrix(10, 10);
            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var indices = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            var actual = sut.Calculate(() => new CrossValidationTestLearner(indices),
                observations, targets);

            var expected = new List<BiasVarianceLearningCurvePoint>() { new BiasVarianceLearningCurvePoint(1, 81, 8), 
                new BiasVarianceLearningCurvePoint(6, 24.166666666666664, 14.5)};
            
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void BiasVarianceLearningCurvesCalculator_Calculate_Metric_Null()
        {
            new BiasVarianceLearningCurvesCalculator(
                null, new double[] { 0.2, 0.8 }, 0.8, 42);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void BiasVarianceLearningCurvesCalculator_Calculate_Sample_Percentages_Null()
        {
            new BiasVarianceLearningCurvesCalculator(
                new CrossValidationTestMetric(), null, 0.8, 42);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BiasVarianceLearningCurvesCalculator_Calculate_Sample_Percentages_Empty()
        {
            new BiasVarianceLearningCurvesCalculator(
                new CrossValidationTestMetric(), new double[] { }, 0.8, 42);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BiasVarianceLearningCurvesCalculator_Calculate_Training_Percentage_Too_Low()
        {
            new BiasVarianceLearningCurvesCalculator(
                new CrossValidationTestMetric(), new double[] { 0.2, 0.8 }, 0.0, 42);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BiasVarianceLearningCurvesCalculator_Calculate_Training_Percentage_Too_High()
        {
            new BiasVarianceLearningCurvesCalculator(
                new CrossValidationTestMetric(), new double[] { 0.2, 0.8 }, 1.0, 42);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BiasVarianceLearningCurvesCalculator_Calculate_Sample_Percentage_Too_Low()
        {
            var sut = new BiasVarianceLearningCurvesCalculator(
                new CrossValidationTestMetric(), new double[] { 0.0, 0.8 }, 0.8, 42);

            var observations = new F64Matrix(10, 10);
            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var indices = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            sut.Calculate(() => new CrossValidationTestLearner(indices),
                observations, targets);

        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BiasVarianceLearningCurvesCalculator_Calculate_Sample_Percentage_Too_High()
        {
            var sut = new BiasVarianceLearningCurvesCalculator(
                new CrossValidationTestMetric(), new double[] { 1.1, 0.8 }, 0.8, 42);

            var observations = new F64Matrix(10, 10);
            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var indices = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            sut.Calculate(() => new CrossValidationTestLearner(indices),
                observations, targets);
        } 
    }
}
