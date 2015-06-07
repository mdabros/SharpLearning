using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using System.Diagnostics;
using SharpLearning.GradientBoost.Loss;

namespace SharpLearning.GradientBoost.Test.Loss
{
    [TestClass]
    public class GradientBoostSquaredLossTest
    {
        [TestMethod]
        public void GBMSquaredLoss_InitializeLoss()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var sut = new GradientBoostSquaredLoss();

            var actual = sut.InitialLoss(targets, targets.Select(t => true).ToArray());
            Assert.AreEqual(5.0, actual);
        }

        [TestMethod]
        public void GBMSquaredLoss_UpdateResiduals()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 };
            var actual = new double[targets.Length];
            var sut = new GradientBoostSquaredLoss();

            sut.UpdateResiduals(targets, predictions, actual, targets.Select(t => true).ToArray());

            var expected = new double[] { -4, -3, -2, -1, 0, 1, 2, 3, 4 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void GBMSquaredLoss_UpdateResiduals_Indexed()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 };
            var actual = new double[targets.Length];
            var inSample = new bool[] { true, false, true, false, true, false, true, false, true };
            var sut = new GradientBoostSquaredLoss();

            sut.UpdateResiduals(targets, predictions, actual, inSample);

            var expected = new double[] { -4, -3, -2, -1, 0, 1, 2, 3, 4 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void GBMSquaredLoss_UpdatedLeafValue()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 2.0, 1.0, 4.0, 3.0, 4.0, 5.0, 8.0, 9.0, 1.0 };
            var inSample = new bool[] { true, true, true, true, true, true, true, true, true };
            var sut = new GradientBoostSquaredLoss();

            var actual = sut.UpdatedLeafValue(0.0, targets, predictions, inSample);
            Assert.AreEqual(0.0, actual, 0.001);
        }

        [TestMethod]
        public void GBMSquaredLoss_UpdatedLeafValue_Indexed()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 2.0, 1.0, 4.0, 3.0, 4.0, 5.0, 8.0, 9.0, 1.0 };
            var inSample = new bool[] { true, false, true, false, true, false, true, false, true };
            var sut = new GradientBoostSquaredLoss();

            var actual = sut.UpdatedLeafValue(0.0, targets, predictions, inSample);
            Assert.AreEqual(0.0, actual, 0.001);
        }

        [TestMethod]
        public void GBMSquaredLoss_UpdateLeafValues()
        {
            var sut = new GradientBoostSquaredLoss();
            Assert.IsFalse(sut.UpdateLeafValues());
        }
    }
}
