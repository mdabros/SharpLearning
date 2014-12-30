using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.Test;
using System;
using System.Linq;

namespace SharpLearning.CrossValidation.CrossValidators.Test
{
    /// <summary>
    /// Summary description for StratifiedCrossValidationTest
    /// </summary>
    [TestClass]
    public class StratifiedCrossValidationTest
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
            var indices = Enumerable.Range(0, targets.Length).ToArray();

            var sut = new StratifiedCrossValidation<double>(folds, 42);
            var actual = sut.CrossValidate(() => new CrossValidationTestLearner(indices), observations, targets);
            return actual;
        }
    }
}
