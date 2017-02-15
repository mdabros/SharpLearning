using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Loss;
using MathNet.Numerics.LinearAlgebra;

namespace SharpLearning.Neural.Test.Loss
{
    [TestClass]
    public class HingeLossTest
    {
        [TestMethod]
        public void HingeLoss_Loss()
        {
            // example from http://cs231n.github.io/linear-classify/#svmvssoftmax
            var sut = new HingeLoss();
            var targets = Matrix<float>.Build.Dense(1, 3, new float[] { 0, 0, 1 });
            var predictions = Matrix<float>.Build.Dense(1, 3, new float[] { -2.85f, 0.86f, 0.28f });

            var actual = sut.Loss(targets, predictions);
            Assert.AreEqual(1.58, actual, 0.001);
        }

        [TestMethod]
        public void HingeLoss_Loss_2()
        {
            var sut = new HingeLoss();
            var targets = Matrix<float>.Build.Dense(3, 2, new float[] { 1f, 1f, 0f, 0f, 0f, 1, });
            var predictions = Matrix<float>.Build.Dense(3, 2, new float[] { 0.9f, 0.9f, 0.1f, .1f, .1f, .9f });

            var actual = sut.Loss(targets, predictions);
            Assert.AreEqual(0.200000018, actual, 0.001);
        }

    }
}
