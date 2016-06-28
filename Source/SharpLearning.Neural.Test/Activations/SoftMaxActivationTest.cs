using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Activations;
using System.Diagnostics;

namespace SharpLearning.Neural.Test.Activations
{
    [TestClass]
    public class SoftMaxActivationTest
    {
        [TestMethod]
        public void SoftMaxActivation_Activation()
        {
            var actual = Matrix<float>.Build.Dense(1, 3, new float[] { 0.1f, 0.23f, 0.86f });
            var sut = new SoftMaxActivation();
            sut.Activation(actual);
            var expected = Matrix<float>.Build.Dense(1, 3, new float[] { 0.233803f, 0.2662615f, 0.4999354f });

            Assert.AreEqual(expected.ToString(), actual.ToString());
        }
    }
}
