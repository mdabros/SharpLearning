using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Optimization.ParameterSamplers;
using SharpLearning.Optimization.Transforms;

namespace SharpLearning.Optimization.Test.Transforms
{
    [TestClass]
    public class LinearTransformTest
    {
        [TestMethod]
        public void LinearTransform_Transform()
        {
            var sut = new LinearTransform();
            var sampler = new RandomUniform(seed: 32);

            var actual = new double[10];
            for (int i = 0; i < actual.Length; i++)
            {
                actual[i] = sut.Transform(min: 20, max: 200,
                    parameterType: ParameterType.Continuous, sampler: sampler);
            }

            var expected = new double[] { 99.8935983236384, 57.2098020451189, 44.4149092419142, 89.9002946307418, 137.643828772774, 114.250629522954, 63.8914499915631, 109.294177409864, 188.567149950455, 33.2731248034505 };
            ArrayAssert.AssertAreEqual(expected, actual);
        }
    }
}
