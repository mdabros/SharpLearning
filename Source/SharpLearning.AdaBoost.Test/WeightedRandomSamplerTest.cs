using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace SharpLearning.AdaBoost.Test
{
    [TestClass]
    public class WeightedRandomSamplerTest
    {
        [TestMethod]
        public void WeightedRandomSampler_Sample_Weight_10()
        {
            var sut = new WeightedRandomSampler();
            var indices = Enumerable.Range(0, 10).ToArray();
            var weights = new double[] { 1, 1, 1, 1, 1, 10, 10, 10, 10, 10 };

            var actual = new int[indices.Length];
            sut.Sample(indices, weights, actual);
            
            var expected = new int[] { 2, 5, 6, 7, 7, 8, 8, 8, 9, 9 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void WeightedRandomSampler_Sample_Weight_Equal()
        {
            var sut = new WeightedRandomSampler();
            var indices = Enumerable.Range(0, 10).ToArray();
            var weights = new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

            var actual = new int[indices.Length];
            sut.Sample(indices, weights, actual);

            var expected = new int[] { 0, 2, 4, 4, 5, 6, 7, 7, 9, 9 };
            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
