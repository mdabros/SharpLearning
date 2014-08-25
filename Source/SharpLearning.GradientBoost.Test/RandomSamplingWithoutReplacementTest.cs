using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace SharpLearning.GradientBoost.Test
{
    [TestClass]
    public class RandomSamplingWithoutReplacementTest
    {

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RandomSamplingWithoutReplacement_Percentage_Too_Small()
        {
            new RandomSamplingWithoutReplacement(-1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RandomSamplingWithoutReplacement_Percentage_Too_Big()
        {
            new RandomSamplingWithoutReplacement(1.1);
        }


        [TestMethod]
        public void RandomSamplingWithoutReplacement_Sample_100()
        {
            var sut = new RandomSamplingWithoutReplacement(1.0);
            var indices = Enumerable.Range(0, 10).ToArray();

            var actual = new int[indices.Length];
            sut.Sample(indices, ref actual);

            CollectionAssert.AreEqual(indices, actual.OrderBy(v => v).ToArray());
        }

        [TestMethod]
        public void RandomSamplingWithoutReplacement_Sample_50()
        {
            var sut = new RandomSamplingWithoutReplacement(.5);
            var indices = Enumerable.Range(0, 10).ToArray();

            var actual = new int[indices.Length];
            sut.Sample(indices, ref actual);

            var expected = new int[] { 9, 0, 4, 2, 5 };
            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
