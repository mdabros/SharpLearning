using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization.Test.ParameterSamplers
{
    [TestClass]
    public class RandomUniformLinearTest
    {
        [TestMethod]
        public void RandomUniformLinear_Sample()
        {
            var sut = new RandomUniformLinear();

            var random = new Random(32);
            var actual = new double[10];
            for (int i = 0; i < actual.Length; i++)
            {
                actual[i] = sut.Sample(min: 20, max: 200, random: random);
            }

            var expected = new double[] { 99.8935983236384, 57.2098020451189, 44.4149092419142, 89.9002946307418, 137.643828772774, 114.250629522954, 63.8914499915631, 109.294177409864, 188.567149950455, 33.2731248034505 };
            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }
    }
}
