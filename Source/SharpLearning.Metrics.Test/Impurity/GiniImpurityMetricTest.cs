using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Views;
using SharpLearning.Metrics.Impurity;

namespace SharpLearning.Metrics.Test.Impurity
{
    [TestClass]
    public class GiniImpurityMetricTest
    {
        [TestMethod]
        public void GiniImpurityMetric_Impurity()
        {
            var set1 = new double[] { 0, 1, 2, 3, 4, 3, 2, 1, 0 };
            var set2 = new double[] { 1, 1, 1, 1, 2, 2, 2, 2 };
            var set3 = new double[] { 1, 1, 1, 1, 1, 1, 1, 1 };

            var sut = new GiniImpurityMetric();

            var val1 = sut.Impurity(set1);
            Assert.AreEqual(0.79012345679012341, val1);
            var val2 = sut.Impurity(set2);
            Assert.AreEqual(0.5, val2);
            var val3 = sut.Impurity(set3);
            Assert.AreEqual(0.0, val3);
        }

        [TestMethod]
        public void GiniImpurityMetric_Impurity_Interval()
        {
            var set1 = new double[] { 0, 1, 2, 3, 4, 3, 2, 1, 0 };
            var set2 = new double[] { 1, 1, 1, 1, 2, 2, 2, 2 };
            var set3 = new double[] { 1, 1, 1, 1, 1, 1, 1, 1 };

            var sut = new GiniImpurityMetric();
            var interval = Interval1D.Create(2, 7);

            var val1 = sut.Impurity(set1, interval);
            Assert.AreEqual(0.64, val1);
            var val2 = sut.Impurity(set2, interval);
            Assert.AreEqual(0.48, val2);
            var val3 = sut.Impurity(set3, interval);
            Assert.AreEqual(0.0, val3);
        }

        [TestMethod]
        public void GiniImpurityMetric_Impurity_Weights_None()
        {
            var set1 = new double[] { 0, 1, 2, 3, 4, 3, 2, 1, 0 };
            var set2 = new double[] { 1, 1, 1, 1, 2, 2, 2, 2 };
            var set3 = new double[] { 1, 1, 1, 1, 1, 1, 1, 1 };

            var weights = new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
            var sut = new GiniImpurityMetric();

            var val1 = sut.Impurity(set1, weights, Interval1D.Create(0, set1.Length));
            Assert.AreEqual(0.79012345679012341, val1);
            var val2 = sut.Impurity(set2, weights, Interval1D.Create(0, set2.Length));
            Assert.AreEqual(0.5, val2);
            var val3 = sut.Impurity(set3, weights, Interval1D.Create(0, set3.Length));
            Assert.AreEqual(0.0, val3);
        }

        [TestMethod]
        public void GiniImpurityMetric_Impurity_Weights()
        {
            var set1 = new double[] { 0, 1, 2, 3, 4, 3, 2, 1, 0 };
            var set2 = new double[] { 1, 1, 1, 1, 2, 2, 2, 2 };
            var set3 = new double[] { 1, 1, 1, 1, 1, 1, 1, 1 };

            var weights = new double[] { 3, 1, 4, 1, 0.5, 1, 2, 1, 10 };
            var sut = new GiniImpurityMetric();

            var val1 = sut.Impurity(set1, weights, Interval1D.Create(0, set1.Length));
            Assert.AreEqual(0.96921684019918519, val1);
            var val2 = sut.Impurity(set2, weights, Interval1D.Create(0, set2.Length));
            Assert.AreEqual(0.82441700960219477, val2);
            var val3 = sut.Impurity(set3, weights, Interval1D.Create(0, set3.Length));
            Assert.AreEqual(0.64883401920438954, val3);
        }
    }
}
