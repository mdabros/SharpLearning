using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Loss;

namespace SharpLearning.Neural.Test.Loss
{
    [TestClass]
    public class SquaredLossTest
    {
        [TestMethod]
        public void SquaredLoss_Loss()
        {
            var targets = Matrix<float>.Build.Dense(6, 1, new float[] { 0, 0, 0, 0, 0, 0 });
            var predictions = Matrix<float>.Build.Dense(6, 1, new float[] { 0, 0, 0, 0, 0, 0 });

            var sut = new SquaredLoss();
            var actual = sut.Loss(targets, predictions);

            Assert.AreEqual(0f, actual);
        }

        [TestMethod]
        public void SquaredLoss_Loss_1()
        {
            var targets = Matrix<float>.Build.Dense(5, 1, new float[] { 1.0f, 2.3f, 3.1f, 4.4f, 5.8f });
            var predictions = Matrix<float>.Build.Dense(5, 1, new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

            var sut = new SquaredLoss();
            var actual = sut.Loss(targets, predictions);

            Assert.AreEqual(0.18f, actual, 0.0001);
        }
    }
}
