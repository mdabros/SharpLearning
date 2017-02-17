using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Views;
using SharpLearning.Metrics.Impurity;
using System.Linq;

namespace SharpLearning.Metrics.Test.Impurity
{
    [TestClass]
    public class NaiveSinglePassVarianceImpurityMetricTest
    {
        [TestMethod]
        public void NaiveSinglePassVarianceEntropyMetric_Impurity()
        {
            var set1 = new double[] { 0, 1, 2, 3, 4, 3, 2, 1, 0 };
            var set2 = new double[] { 1, 1, 1, 1, 2, 2, 2, 2 };
            var set3 = new double[] { 1 };

            var sut = new NaiveSinglePassVarianceImpurityMetric();

            var val1 = sut.Impurity(set1);
            Assert.AreEqual(1.9444444444444446, val1);
            var val2 = sut.Impurity(set2);
            Assert.AreEqual(0.2857142857142857, val2);
            var val3 = sut.Impurity(set3);
            Assert.AreEqual(0.0, val3);
        }

        [TestMethod]
        public void NaiveSinglePassVarianceEntropyMetric_Impurity_Interval()
        {
            var set1 = new double[] { 0, 1, 2, 3, 4, 3, 2, 1, 0 };
            var set2 = new double[] { 1, 1, 1, 1, 2, 2, 2, 2 };

            var sut = new NaiveSinglePassVarianceImpurityMetric();
            var interval = Interval1D.Create(2, 7);

            var val1 = sut.Impurity(set1, interval);
            Assert.AreEqual(0.69999999999999929, val1);
            var val2 = sut.Impurity(set2, interval);
            Assert.AreEqual(0.29999999999999982, val2);
        }


        [TestMethod]
        public void NaiveSinglePassVarianceEntropyMetric_Impurity_Interval_Weighted_1()
        {
            var set1 = new double[] { 0, 1, 2, 3, 4, 3, 2, 1, 0 };
            var set2 = new double[] { 1, 1, 1, 1, 2, 2, 2, 2 };
            var set3 = new double[] { 1 };

            var weights = new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1 };

            var sut = new NaiveSinglePassVarianceImpurityMetric();
            var interval = Interval1D.Create(2, 7);

            var val1 = sut.Impurity(set1, weights, Interval1D.Create(0, set1.Length));
            Assert.AreEqual(1.9444444444444446, val1);
            var val2 = sut.Impurity(set2, weights, Interval1D.Create(0, set2.Length));
            Assert.AreEqual(0.2857142857142857, val2);
            var val3 = sut.Impurity(set3, weights, Interval1D.Create(0, set3.Length));
            Assert.AreEqual(0.0, val3);
        }

        [TestMethod]
        public void NaiveSinglePassVarianceEntropyMetric_Impurity_Interval_Weighted_2()
        {
            var set1 = new double[] { 0, 1, 2, 3, 4, 3, 2, 1, 0 };
            var set2 = new double[] { 1, 1, 1, 1, 2, 2, 2, 2 };
            var set3 = new double[] { 1 };

            var weights = new double[] { 1, 2, 4, 7, 2, 3, 5, 8, 1 };

            var sut = new NaiveSinglePassVarianceImpurityMetric();
            var interval = Interval1D.Create(2, 7);

            var val1 = sut.Impurity(set1, weights, Interval1D.Create(0, set1.Length));
            Assert.AreEqual(1.2969432314410481, val1);
            var val2 = sut.Impurity(set2, weights, Interval1D.Create(0, set2.Length));
            Assert.AreEqual(0.29577464788732394, val2);
            var val3 = sut.Impurity(set3, weights, Interval1D.Create(0, set3.Length));
            Assert.AreEqual(0.0, val3);
        }

    }
}
