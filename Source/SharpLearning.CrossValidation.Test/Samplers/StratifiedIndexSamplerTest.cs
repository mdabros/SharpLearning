using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation.Test.Samplers
{
    [TestClass]
    public class StratifiedIndexSamplerTest
    {
        [TestMethod]
        public void StratifiedIndexSampler_Sample()
        {
            var values = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            var sampleSize = values.Length / 2;
            var sut = new StratifiedIndexSampler<int>();
            var sampleIndices = sut.Sample(values, sampleSize);

            var actual = values.GetIndices(sampleIndices);
            var expected = new int[] { 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3 };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void StratifiedIndexSampler_Sample_Indexed()
        {
            var values = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            var indices = new int[] { 0, 1, 2, 3, 10, 11, 12, 13, 18, 19, 20 };
            var sampleSize = 6;
            var sut = new StratifiedIndexSampler<int>();

            var sampleIndices = sut.Sample(values, sampleSize, indices);

            var actual = values.GetIndices(sampleIndices);
            var expected = new int[] { 1, 1, 1, 2, 2, 3 };

            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
