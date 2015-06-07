using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using System.Diagnostics;
using SharpLearning.GradientBoost.LossFunctions;

namespace SharpLearning.GradientBoost.Test.LossFunctions
{
    [TestClass]
    public class GBMQuantileLossTest
    {
        [TestMethod]
        public void GBMQuantileLoss_InitializeLoss()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var sut = new GBMQuantileLoss(0.9);

            var actual = sut.InitialLoss(targets, targets.Select(t => true).ToArray());
            Assert.AreEqual(8.2, actual);
        }

        [TestMethod]
        public void GBMQuantileLoss_UpdateResiduals()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 };
            var actual = new double[targets.Length];
            var sut = new GBMQuantileLoss(0.9);

            sut.UpdateResiduals(targets, predictions, actual, targets.Select(t => true).ToArray());

            var expected = new double[] { -0.1, -0.1, -0.1, -0.1, -0.1, 0.9, 0.9, 0.9, 0.9 };

            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.001);
            }
        }

        [TestMethod]
        public void GBMQuantileLoss_UpdateResiduals_Indexed()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 };
            var actual = new double[targets.Length];
            var inSample = new bool[] { true, false, true, false, true, false, true, false, true };
            var sut = new GBMQuantileLoss(0.9);

            sut.UpdateResiduals(targets, predictions, actual, inSample);

            var expected = new double[] { -0.1, -0.1, -0.1, -0.1, -0.1, 0.9, 0.9, 0.9, 0.9 };

            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.001);
            }
        }

        [TestMethod]
        public void GBMQuantileLoss_UpdatedLeafValue()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 2.0, 1.0, 4.0, 3.0, 4.0, 5.0, 8.0, 9.0, 1.0 };
            var inSample = new bool[] { true, true, true, true, true, true, true, true, true };
            var sut = new GBMQuantileLoss(0.9);

            var actual = sut.UpdatedLeafValue(0.0, targets, predictions, inSample);
            Assert.AreEqual(2.4000000000000012, actual, 0.001);
        }

        [TestMethod]
        public void GBMQuantileLoss_UpdatedLeafValue_Indexed()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 2.0, 1.0, 4.0, 3.0, 4.0, 5.0, 8.0, 9.0, 1.0 };
            var inSample = new bool[] { true, false, true, false, true, false, true, false, true };
            var sut = new GBMQuantileLoss(0.9);

            var actual = sut.UpdatedLeafValue(0.0, targets, predictions, inSample);
            Assert.AreEqual(5.2000000000000011, actual, 0.001);
        }

        [TestMethod]
        public void GBMQuantileLoss_UpdateLeafValues()
        {
            var sut = new GBMQuantileLoss(0.9);
            Assert.IsTrue(sut.UpdateLeafValues());
        }
    }
}
