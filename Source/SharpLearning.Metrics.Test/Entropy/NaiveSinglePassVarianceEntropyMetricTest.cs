using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Views;
using SharpLearning.Metrics.Entropy;

namespace SharpLearning.Metrics.Test.Entropy
{
    [TestClass]
    public class NaiveSinglePassVarianceEntropyMetricTest
    {
        [TestMethod]
        public void NaiveSinglePassVarianceEntropyMetric_Entropy()
        {
            var set1 = new double[] { 0, 1, 2, 3, 4, 3, 2, 1, 0 };
            var set2 = new double[] { 1, 1, 1, 1, 2, 2, 2, 2 };
            var set3 = new double[] { 1 };

            var sut = new NaiveSinglePassVarianceEntropyMetric();

            var val1 = sut.Entropy(set1);
            Assert.AreEqual(1.9444444444444446, val1);
            var val2 = sut.Entropy(set2);
            Assert.AreEqual(0.2857142857142857, val2);
            var val3 = sut.Entropy(set3);
            Assert.AreEqual(0.0, val3);
        }

        [TestMethod]
        public void NaiveSinglePassVarianceEntropyMetric_Entropy_Interval()
        {
            var set1 = new double[] { 0, 1, 2, 3, 4, 3, 2, 1, 0 };
            var set2 = new double[] { 1, 1, 1, 1, 2, 2, 2, 2 };
            var set3 = new double[] { 1 };

            var sut = new NaiveSinglePassVarianceEntropyMetric();
            var interval = Interval1D.Create(2, 7);

            var val1 = sut.Entropy(set1, interval);
            Assert.AreEqual(0.69999999999999929, val1);
            var val2 = sut.Entropy(set2, interval);
            Assert.AreEqual(0.29999999999999982, val2);
        }
    }
}
