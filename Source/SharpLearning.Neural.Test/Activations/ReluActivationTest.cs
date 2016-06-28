using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using System.Diagnostics;

namespace SharpLearning.Neural.Test.Activations
{
    [TestClass]
    public class ReluActivationTest
    {
        [TestMethod]
        public void ReluActivation_Activiation()
        {
            var actual = Matrix<float>.Build.Random(5, 5, 23);
            var sut = new ReluActivation();
            sut.Activation(actual);

            var expected = Matrix<float>.Build.Dense(5, 5, new float[] { 1.602584f, 0.4367149f, 1.206761f, 0f, 0.9932324f, 1.745592f, 0.6927068f, 0.546773f, 0f, 0f, 0f, 0f, 0f, 0f, 0.2088782f, 0f, 0f, 0f, 1.082098f, 0f, 0.2817991f, 0f, 0.1463978f, 0.3077979f, 0.8454311f });

            Assert.AreEqual(expected.ToString(), actual.ToString());
        }

        [TestMethod]
        public void ReluActivation_Derivative()
        {
            var actual = Matrix<float>.Build.Random(5, 5, 23);
            var sut = new ReluActivation();
            sut.Derivative(actual, actual);

            Trace.WriteLine(string.Join(", ", actual.ToColumnWiseArray()));
            var expected = Matrix<float>.Build.Dense(5, 5, new float[] { 1f, 1f, 1f, 0f, 1f, 1f, 1f, 1f, 0f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f, 1f, 0f, 1f, 1f, 1 });

            Assert.AreEqual(expected.ToString(), actual.ToString());
        }
    }
}
