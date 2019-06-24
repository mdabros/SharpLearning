using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.GradientBoost.Loss;

namespace SharpLearning.GradientBoost.Test.Loss
{
    [TestClass]
    public class GradientBoostBinomialLossTest
    {
        [TestMethod]
        public void GBMBinomialLoss_InitializeLoss()
        {
            var targets = new double[] { 1, 1, 1, 1, 0, 0, 0, 0, 0 };
            var sut = new GradientBoostBinomialLoss();

            var actual = sut.InitialLoss(targets, targets.Select(t => true).ToArray());
            Assert.AreEqual(-0.22314355131420971, actual, 0.001);
        }

        [TestMethod]
        public void GBMBinomialLoss_UpdateResiduals()
        {
            var targets = new double[] { 1, 1, 1, 1, 0, 0, 0, 0, 0 };
            var predictions = new double[] { 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 };
            var actual = new double[targets.Length];
            var sut = new GradientBoostBinomialLoss();

            sut.UpdateResiduals(targets, predictions, actual, targets.Select(t => true).ToArray());

            var expected = new double[] { 0.268941421369995, 0.268941421369995, 0.5, 0.268941421369995, -0.731058578630005, -0.731058578630005, -0.5, -0.5, -0.5 };

            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.001);
            }
        }

        [TestMethod]
        public void GBMBinomialLoss_UpdateResiduals_Indexed()
        {
            var targets = new double[] { 1, 1, 1, 1, 0, 0, 0, 0, 0 };
            var predictions = new double[] { 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 }; 
            var actual = new double[targets.Length];
            var inSample = new bool[] { true, false, true, false, true, false, true, false, true };
            var sut = new GradientBoostBinomialLoss();

            sut.UpdateResiduals(targets, predictions, actual, inSample);

            var expected = new double[] { 0.268941421369995, 0.0, 0.5, 0.0, -0.731058578630005, 0.0, -0.5, 0.0, -0.5 };

            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.001);
            }
        }

        [TestMethod]
        public void GBMBinomialLoss_UpdatedLeafValue()
        {
            var targets = new double[] { 1, 1, 1, 1, 0, 0, 0, 0, 0 };
            var predictions = new double[] { 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 }; 
            var inSample = new bool[] { true, true, true, true, true, true, true, true, true };
            var sut = new GradientBoostBinomialLoss();

            var actual = sut.UpdatedLeafValue(0.0, targets, predictions, inSample);
            Assert.AreEqual(0.0, actual, 0.001);
        }

        [TestMethod]
        public void GBMBinomialLoss_UpdatedLeafValue_Indexed()
        {
            var targets = new double[] { 1, 1, 1, 1, 0, 0, 0, 0, 0 };
            var predictions = new double[] { 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 }; 
            var inSample = new bool[] { true, false, true, false, true, false, true, false, true };
            var sut = new GradientBoostBinomialLoss();

            var actual = sut.UpdatedLeafValue(0.0, targets, predictions, inSample);
            Assert.AreEqual(0.0, actual, 0.001);
        }

        [TestMethod]
        public void GBMBinomialLoss_UpdateLeafValues()
        {
            var sut = new GradientBoostBinomialLoss();
            Assert.IsFalse(sut.UpdateLeafValues());
        }
    }
}
