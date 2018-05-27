using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization.Test.ParameterSamplers
{
    [TestClass]
    public class RandomUniformIntergersTest
    {
        [TestMethod]
        public void RandomUniformIntergers_Sample()
        {
            var sut = new RandomUniformIntergers(32);

            var actual = new double[10];
            for (int i = 0; i < actual.Length; i++)
            {
                actual[i] = sut.Sample(min: 20, max: 200);
            }

            var expected = new double[] { 100, 57, 44, 90, 138, 114, 64, 109, 189, 33 };
            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RandomUniformIntergers_Throw_On_Min_Larger_Than_Max()
        {
            var sut = new RandomUniformIntergers(32);
            sut.Sample(min: 20, max: 10);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RandomUniformIntergers_Throw_On_Min_Equals_Than_Max()
        {
            var sut = new RandomUniformIntergers(32);
            sut.Sample(min: 20, max: 20);
        }
    }
}
