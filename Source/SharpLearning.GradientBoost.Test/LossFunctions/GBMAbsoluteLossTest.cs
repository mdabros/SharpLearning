using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using SharpLearning.GradientBoost.GBM;

namespace SharpLearning.GradientBoost.Test.GBM.LossFunctions
{
    [TestClass]
    public class GBMAbsoluteLossTest
    {
        [TestMethod]
        public void GBMAbsoluteLoss_InitializeLoss()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var sut = new GBMAbsoluteLoss();

            var actual = sut.InitialLoss(targets, targets.Select(t => true).ToArray());
            Assert.AreEqual(5.0, actual);
        }

        [TestMethod]
        public void GBMAbsoluteLoss_UpdateResiduals()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 };
            var actual = new double[targets.Length];
            var sut = new GBMAbsoluteLoss();

            sut.UpdateResiduals(targets, predictions, actual, targets.Select(t => true).ToArray());

            var expected = new double[] { -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void GBMAbsoluteLoss_UpdateResiduals_Indexed()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 };
            var actual = new double[targets.Length];
            var inSample = new bool[] { true, false, true, false, true, false, true, false, true };
            var sut = new GBMAbsoluteLoss();

            sut.UpdateResiduals(targets, predictions, actual, inSample);

            var expected = new double[] { -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void GBMAbsoluteLoss_UpdatedLeafValue()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 2.0, 1.0, 4.0, 3.0, 4.0, 5.0, 8.0, 9.0, 1.0 };
            var inSample = new bool[] { true, true, true, true, true, true, true, true, true };
            var sut = new GBMAbsoluteLoss();

            var actual = sut.UpdatedLeafValue(0.0, targets, predictions, inSample);
            Assert.AreEqual(1.0, actual, 0.001);
        }

        [TestMethod]
        public void GBMAbsoluteLoss_UpdatedLeafValue_Indexed()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 2.0, 1.0, 4.0, 3.0, 4.0, 5.0, 8.0, 9.0, 1.0 };
            var inSample = new bool[] { true, false, true, false, true, false, true, false, true };
            var sut = new GBMAbsoluteLoss();

            var actual = sut.UpdatedLeafValue(0.0, targets, predictions, inSample);
            Assert.AreEqual(-1.0, actual, 0.001);
        }

        [TestMethod]
        public void GBMAbsoluteLoss_UpdateLeafValues()
        {
            var sut = new GBMAbsoluteLoss();
            Assert.IsTrue(sut.UpdateLeafValues());
        }
    }
}
