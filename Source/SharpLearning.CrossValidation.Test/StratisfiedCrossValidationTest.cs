using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using System.Linq;

namespace SharpLearning.CrossValidation.Test
{
    /// <summary>
    /// Summary description for StratisfiedCrossValidationTest
    /// </summary>
    [TestClass]
    public class StratisfiedCrossValidationTest
    {
        [TestMethod]
        public void StratisfiedCrossValidation_CrossValidate_Folds_2()
        {
            var actual = AssertCrossValidation(2);
            var expected = new double[] { 5, 8, 3, 6, 7, 4, 2, 9, 1, 0 };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void StratisfiedCrossValidation_CrossValidate_Folds_4()
        {
            var actual = AssertCrossValidation(4);
            var expected = new double[] { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void StratisfiedCrossValidation_CrossValidate_Too_Many_Folds()
        {
            AssertCrossValidation(20);
        }

        double[] AssertCrossValidation(int folds)
        {
            var observations = new F64Matrix(10, 10);
            var targets = new double[] { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };

            var sut = new StratisfiedCrossValidation<double, double>(ModelLearner, folds, 42);
            var actual = sut.CrossValidate(observations, targets);
            return actual;
        }

        CrossValidationEvaluator<double> ModelLearner(F64Matrix observations, double[] targets, int[] foldIndices)
        {
            var indices = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var holdOut = indices.Except(foldIndices).ToArray();

            return (o, s) => holdOut.Select(i => (double)i).ToArray();
        }
    }
}
