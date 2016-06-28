using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Activations;
using MathNet.Numerics.LinearAlgebra;

namespace SharpLearning.Neural.Test.Activations
{
    [TestClass]
    public class IdentityActivationTest
    {
        [TestMethod]
        public void IdentityActivation_Activation()
        {
            var actual = Matrix<float>.Build.Random(5, 5, 23);
            var expected = Matrix<float>.Build.Dense(5, 5);
            actual.CopyTo(expected);

            var sut = new IdentityActivation();
            sut.Activation(actual);

            Assert.AreEqual(expected.ToString(), actual.ToString());
        }
    }
}
