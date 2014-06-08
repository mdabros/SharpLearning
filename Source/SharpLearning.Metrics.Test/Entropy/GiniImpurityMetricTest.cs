using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Views;
using SharpLearning.Metrics.Entropy;

namespace SharpLearning.Metrics.Test.Entropy
{
    [TestClass]
    public class GiniImpurityMetricTest
    {
        [TestMethod]
        public void GiniImpurityMetric_Entropy()
        {
            var set1 = new double[] { 0, 1, 2, 3, 4, 3, 2, 1, 0 };
            var set2 = new double[] { 1, 1, 1, 1, 2, 2, 2, 2 };
            var set3 = new double[] { 1, 1, 1, 1, 1, 1, 1, 1 };

            var sut = new GiniImpurityMetric();

            var val1 = sut.Entropy(set1);
            Assert.AreEqual(0.79012345679012341, val1);
            var val2 = sut.Entropy(set2);
            Assert.AreEqual(0.5, val2);
            var val3 = sut.Entropy(set3);
            Assert.AreEqual(0.0, val3);
        }

        [TestMethod]
        public void GiniImpurityMetric_Entropy_Interval()
        {
            var set1 = new double[] { 0, 1, 2, 3, 4, 3, 2, 1, 0 };
            var set2 = new double[] { 1, 1, 1, 1, 2, 2, 2, 2 };
            var set3 = new double[] { 1, 1, 1, 1, 1, 1, 1, 1 };

            var sut = new GiniImpurityMetric();
            var interval = Interval1D.Create(2, 7);

            var val1 = sut.Entropy(set1, interval);
            Assert.AreEqual(0.6399999999999999, val1);
            var val2 = sut.Entropy(set2, interval);
            Assert.AreEqual(0.47999999999999987, val2);
            var val3 = sut.Entropy(set3, interval);
            Assert.AreEqual(0.0, val3);
        }

    }
}
