using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class ParameterBoundsTest
    {
        [TestMethod]
        public void ParameterBounds_NextValue()
        {
            var sut = new MinMaxParameterSpec(min: 20, max: 200, 
                transform: Transform.Linear);

            var sampler = new RandomUniform(seed: 32);

            var actual = new double[10];
            for (int i = 0; i < actual.Length; i++)
            {
                actual[i] = sut.SampleValue(sampler: sampler);
            }

            var expected = new double[] 
            {
                99.8935983236384,
                57.2098020451189,
                44.4149092419142,
                89.9002946307418,
                137.643828772774,
                114.250629522954,
                63.8914499915631,
                109.294177409864,
                188.567149950455,
                33.2731248034505
            };

            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ParameterBounds_Throw_On_Min_Larger_Than_Max()
        {
            new MinMaxParameterSpec(min:30, max: 10);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ParameterBounds_Throw_On_Min_Equals_Than_Max()
        {
            new MinMaxParameterSpec(min: 30, max: 30);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ParameterBounds_Throw_On_Transform_Is_Null()
        {
            new MinMaxParameterSpec(min: 10, max: 30, transform: null, 
                parameterType: ParameterType.Continuous);
        }
    }
}
