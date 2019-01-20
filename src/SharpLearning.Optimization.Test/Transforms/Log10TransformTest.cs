using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Optimization.ParameterSamplers;
using SharpLearning.Optimization.Transforms;

namespace SharpLearning.Optimization.Test.Transforms
{
    [TestClass]
    public class Log10TransformTest
    {
        [TestMethod]
        public void Log10Transform_Transform()
        {
            var sut = new Log10Transform();
            var sampler = new RandomUniform(seed: 32);

            var actual = new double[10];
            for (int i = 0; i < actual.Length; i++)
            {
                actual[i] = sut.Transform(min: 0.0001, max: 1, 
                    parameterType: ParameterType.Continuous, sampler: sampler);
            }

            var expected = new double[] { 0.00596229274859676, 0.000671250295495889, 0.000348781578382963, 0.00357552550811494, 0.0411440752926137, 0.012429636665806, 0.000944855847942692, 0.00964528475124291, 0.557104498829374, 0.000197223348905772, };
            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Log10Transform_Throw_On_Min_equal_Zero()
        {
            var sut = new Log10Transform();
            var sampler = new RandomUniform(seed: 32);
            sut.Transform(min: 0, max: 1,
                parameterType: ParameterType.Continuous, sampler: sampler);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Log10Transform_Throw_On_Max_equal_Zero()
        {
            var sut = new Log10Transform();
            var sampler = new RandomUniform(seed: 32);
            sut.Transform(min: 0.01, max: 0,
                parameterType: ParameterType.Continuous, sampler: sampler);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Log10Transform_Throw_On_Min_below_Zero()
        {
            var sut = new Log10Transform();
            var sampler = new RandomUniform(seed: 32);
            sut.Transform(min: -0.1, max: 1,
                parameterType: ParameterType.Continuous, sampler: sampler);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Log10Transform_Throw_On_Max_below_Zero()
        {
            var sut = new Log10Transform();
            var sampler = new RandomUniform(seed: 32);
            sut.Transform(min: 0.1, max: -0.1,
                parameterType: ParameterType.Continuous, sampler: sampler);
        }
    }
}
