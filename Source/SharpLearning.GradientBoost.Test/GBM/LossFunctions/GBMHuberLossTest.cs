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
    }
}
