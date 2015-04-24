using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.CrossValidation.Samplers;
using System;
using System.Diagnostics;

namespace SharpLearning.CrossValidation.Test.Samplers
{
    [TestClass]
    public class RandomIndexSamplerTest
    {
        [TestMethod]
        public void RandomIndexSampler_Sample()
        {
            var values = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            var sampleSize = values.Length / 2;
            var sut = new RandomIndexSampler<int>(42);
            var sampleIndices = sut.Sample(values, sampleSize);

            var actual = values.GetIndices(sampleIndices);
            var expected = new int[] { 1, 2, 1, 2, 1, 3, 1, 2, 2, 1, 2, };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RandomIndexSampler_Sample_SampleSize_Too_Large()
        {
            var values = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            var sampleSize = 100;
            var sut = new RandomIndexSampler<int>(42);
            var sampleIndices = sut.Sample(values, sampleSize);
        }

        [TestMethod]
        public void RandomIndexSampler_Sample_Indexed()
        {
            var values = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            var indices = new int[] { 0, 1, 2, 3, 10, 11, 12, 13, 18, 19, 20 };
            var sampleSize = 6;
            var sut = new RandomIndexSampler<int>(42);

            var sampleIndices = sut.Sample(values, sampleSize, indices);

            var actual = values.GetIndices(sampleIndices);
            var expected = new int[] { 3, 2, 1, 1, 1, 2 };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RandomIndexSampler_Sample_Indexed_SampleSize_Too_Large()
        {
            var values = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            var indices = new int[] { 0, 1, 2, 3, 10, 11, 12, 13, 18, 19, 20 };
            var sampleSize = 100;
            var sut = new RandomIndexSampler<int>(42);

            var sampleIndices = sut.Sample(values, sampleSize, indices);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RandomIndexSampler_Sample_Indexed_DataIndices_Too_Large()
        {
            var values = new int[] { 1, 1, 1 };
            var indices = new int[] { 0, 1, 2, 3, 10, 11, 12, 13, 18, 19, 20 };
            var sampleSize = 2;
            var sut = new RandomIndexSampler<int>(42);

            var sampleIndices = sut.Sample(values, sampleSize, indices);
        }
    }
}
