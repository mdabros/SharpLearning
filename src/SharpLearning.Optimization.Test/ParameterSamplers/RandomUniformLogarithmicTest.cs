using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization.Test.ParameterSamplers
{
    [TestClass]
    public class RandomUniformLogarithmicTest
    {
        [TestMethod]
        public void RandomUniformLogarithmic_Sample()
        {
            var sut = new RandomUniformLogarithmic();

            var random = new Random(32);
            var actual = new double[10];
            for (int i = 0; i < actual.Length; i++)
            {
                actual[i] = sut.Sample(min: 0.0001, max: 1, random: random);
            }

            var expected = new double[] { 0.00596229274859676, 0.000671250295495889, 0.000348781578382963, 0.00357552550811494, 0.0411440752926137, 0.012429636665806, 0.000944855847942692, 0.00964528475124291, 0.557104498829374, 0.000197223348905772, };
            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }
    }
}
