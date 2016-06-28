using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using System.Diagnostics;

namespace SharpLearning.Neural.Test.Activations
{
    [TestClass]
    public class LogisticActiviationTestcs
    {
        [TestMethod]
        public void LogisticActiviation_Activation()
        {
            var actual = Matrix<float>.Build.Random(5, 5, 23);
            var sut = new LogisticActiviation();
            sut.Activation(actual);

            Trace.WriteLine(string.Join(", ", actual.ToColumnWiseArray()));
            var expected = Matrix<float>.Build.Dense(5, 5, new float[] { 0.8323793f, 0.607476f, 0.7697253f, 0.4714576f, 0.7297259f, 0.851396f, 0.6665688f, 0.6333866f, 0.1840156f, 0.3938053f, 0.1492871f, 0.118243f, 0.322216f, 0.4532051f, 0.5520305f, 0.1586165f, 0.3291628f, 0.2699874f, 0.7468907f, 0.4933237f, 0.5699872f, 0.4720235f, 0.5365342f, 0.5763476f, 0.6996078f });

            Assert.AreEqual(expected.ToString(), actual.ToString());
        }

        [TestMethod]
        public void LogisticActiviation_Derivative()
        {
            var actual = Matrix<float>.Build.Random(5, 5, 23);
            var sut = new LogisticActiviation();
            sut.Derivative(actual, actual);

            var expected = Matrix<float>.Build.Dense(5, 5, new float[] { -0.965692f, 0.245995f, -0.2495108f, -0.1273571f, 0.00672183f, -1.3015f, 0.2128641f, 0.2478123f, -3.707613f, -0.6174028f, -4.768512f, -6.045957f, -1.296557f, -0.2229711f, 0.1652481f, -4.452646f, -1.218881f, -1.984088f, -0.08883768f, -0.02742016f, 0.2023884f, -0.124572f, 0.1249655f, 0.2130583f, 0.1306773f });

            Assert.AreEqual(expected.ToString(), actual.ToString());
        }
    }
}
