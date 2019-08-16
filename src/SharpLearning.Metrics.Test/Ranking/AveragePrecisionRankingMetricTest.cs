using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Ranking;

namespace SharpLearning.Metrics.Test.Ranking
{
    [TestClass]
    public class AveragePrecisionRankingMetricTest
    {
        [TestMethod]
        public void AveragePrecisionRankingMetric_Error_Top_3_1()
        {
            var targets = new int[] { 1, 2, 3 };
            var predictions = new int[] { 1, 2, 1 };
            var sut = new AveragePrecisionRankingMetric<int>(3);

            var actual = sut.Error(targets, predictions);
            Assert.AreEqual(1.0 - 2.0 / 3.0, actual, 0.0001);
        }

        [TestMethod]
        public void AveragePrecisionRankingMetric_Error_Top_3_2()
        {
            var targets = new int[] { 1, 2, 3 };
            var predictions = new int[] { 1, 1, 1 };
            var sut = new AveragePrecisionRankingMetric<int>(3);

            var actual = sut.Error(targets, predictions);
            Assert.AreEqual(1.0 - 1.0 / 3.0, actual, 0.0001);
        }

        [TestMethod]
        public void AveragePrecisionRankingMetric_Error_Top_3_3()
        {
            var targets = new int[] { 1, 3 };
            var predictions = new int[] { 1, 2, 3, 4, 5 };
            var sut = new AveragePrecisionRankingMetric<int>(3);

            var actual = sut.Error(targets, predictions);
            Assert.AreEqual(1.0 - 5.0 / 6.0, actual, 0.0001);
        }

        [TestMethod]
        public void AveragePrecisionRankingMetric_Error_Top_20()
        {
            var targets = Enumerable.Range(1, 100).ToArray();
            var first = Enumerable.Range(1, 20).ToList();
            first.AddRange(Enumerable.Range(200, 600));
            var predictions = first.ToArray();
            var sut = new AveragePrecisionRankingMetric<int>(20);

            var actual = sut.Error(targets, predictions);
            Assert.AreEqual(1.0 - 1.0, actual, 0.0001);
        }

        [TestMethod]
        public void AveragePrecisionRankingMetric_Error_Top_2()
        {
            var targets = Enumerable.Range(1, 5).ToArray();
            var predictions = new int[] { 6, 4, 7, 1, 2 };
            var sut = new AveragePrecisionRankingMetric<int>(2);

            var actual = sut.Error(targets, predictions);
            Assert.AreEqual(1.0 - 0.25, actual, 0.0001);
        }

        [TestMethod]
        public void AveragePrecisionRankingMetric_Error_Top_5()
        {
            var targets = Enumerable.Range(1, 5).ToArray();
            var predictions = new int[] { 1, 1, 1, 1, 1 };
            var sut = new AveragePrecisionRankingMetric<int>(5);

            var actual = sut.Error(targets, predictions);
            Assert.AreEqual(1.0 - 0.2, actual, 0.0001);
        }
    }
}
