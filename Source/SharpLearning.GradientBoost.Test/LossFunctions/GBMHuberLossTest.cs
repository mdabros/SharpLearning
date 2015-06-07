using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using SharpLearning.GradientBoost.GBM;

namespace SharpLearning.GradientBoost.Test.GBM.LossFunctions
{
    [TestClass]
    public class GBMHuberLossTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void GBMHuberLoss_Alpha_Parameter_Too_Low()
        {
            new GBMHuberLoss(0.0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void GBMHuberLoss_Alpha_Parameter_Too_High()
        {
            new GBMHuberLoss(1.1);
        }

        [TestMethod]
        public void GBMHuberLoss_InitializeLoss()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var sut = new GBMHuberLoss(0.9);

            var actual = sut.InitialLoss(targets, targets.Select(t => true).ToArray());
            Assert.AreEqual(5.0, actual);
        }

        [TestMethod]
        public void GBMHuberLoss_UpdateResiduals()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 };
            var actual = new double[targets.Length];
            var sut = new GBMHuberLoss(0.9);

            sut.UpdateResiduals(targets, predictions, actual, targets.Select(t => true).ToArray());

            var expected = new double[] { -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void GBMHuberLoss_UpdateResiduals_Indexed()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 };
            var actual = new double[targets.Length];
            var inSample = new bool[] { true, false, true, false, true, false, true, false, true };
            var sut = new GBMHuberLoss(0.9);

            sut.UpdateResiduals(targets, predictions, actual, inSample);

            var expected = new double[] { -4.0, 0.0, -2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 4.0 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void GBMHuberLoss_UpdatedLeafValue()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 2.0, 1.0, 4.0, 3.0, 4.0, 5.0, 8.0, 9.0, 1.0 };
            var residals = new double[targets.Length];
            var inSample = new bool[] { true, true, true, true, true, true, true, true, true };
            var sut = new GBMHuberLoss(0.9);

            sut.UpdateResiduals(targets, predictions, residals, inSample);

            var actual = sut.UpdatedLeafValue(0.0, targets, predictions, inSample);
            Assert.AreEqual(0.37777777777777788, actual, 0.001);
        }

        [TestMethod]
        public void GBMHuberLoss_UpdatedLeafValue_Indexed()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 2.0, 1.0, 4.0, 3.0, 4.0, 5.0, 8.0, 9.0, 1.0 };
            var residals = new double[targets.Length];
            var inSample = new bool[] { true, false, true, false, true, false, true, false, true };
            var sut = new GBMHuberLoss(0.9);

            sut.UpdateResiduals(targets, predictions, residals, inSample);

            var actual = sut.UpdatedLeafValue(0.0, targets, predictions, inSample);
            Assert.AreEqual(-0.11999999999999977, actual, 0.001);
        }

        [TestMethod]
        public void GBMHuberLoss_UpdateLeafValues()
        {
            var sut = new GBMHuberLoss(0.9);
            Assert.IsTrue(sut.UpdateLeafValues());
        }
    }
}
