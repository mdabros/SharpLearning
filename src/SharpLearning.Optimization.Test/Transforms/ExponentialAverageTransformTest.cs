using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Optimization.ParameterSamplers;
using SharpLearning.Optimization.Transforms;

namespace SharpLearning.Optimization.Test.Transforms
{
    [TestClass]
    public class ExponentialAverageTransformTest
    {
        [TestMethod]
        public void ExponentialAverageTransform_Transform()
        {
            var sut = new ExponentialAverageTransform();
            var sampler = new RandomUniform(seed: 32);

            var actual = new double[10];
            for (int i = 0; i < actual.Length; i++)
            {   
                actual[i] = sut.Transform(min: 0.9, max: 0.999, 
                    parameterType: ParameterType.Continuous, sampler: sampler);
            }

            var expected = new double[] { 0.992278411595665, 0.997409150148125, 0.998132430514324, 0.994020430192635, 0.979715997610774, 0.988851171960333, 0.996926149242493, 0.990178958939479, 0.925360566800827, 0.998595637693094 };
            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ExponentialAverageTransform_Throw_On_Min_equal_One()
        {
            var sut = new ExponentialAverageTransform();
            var sampler = new RandomUniform(seed: 32);
            sut.Transform(min: 1.0, max: 0.5,
                parameterType: ParameterType.Continuous, sampler: sampler);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ExponentialAverageTransform_Throw_On_Max_equal_One()
        {
            var sut = new ExponentialAverageTransform();
            var sampler = new RandomUniform(seed: 32);
            sut.Transform(min: 0.9, max: 1.0,
                parameterType: ParameterType.Continuous, sampler: sampler);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ExponentialAverageTransform_Throw_On_Min_Larger_Than_One()
        {
            var sut = new ExponentialAverageTransform();
            var sampler = new RandomUniform(seed: 32);
            sut.Transform(min: 1.1, max: 0.99,
                parameterType: ParameterType.Continuous, sampler: sampler);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ExponentialAverageTransform_Throw_On_Max_Larger_Than_One()
        {
            var sut = new ExponentialAverageTransform();
            var sampler = new RandomUniform(seed: 32);
            sut.Transform(min: 0.99, max: 1.1,
                parameterType: ParameterType.Continuous, sampler: sampler);
        }
    }
}
