using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Loss;

namespace SharpLearning.Neural.Test.Loss
{
    /// <summary>
    /// Summary description for LogLossTest
    /// </summary>
    [TestClass]
    public class LogLossTest
    {
        [TestMethod]
        public void LogLoss_Loss()
        {
            var sut = new LogLoss();
            var targets = Matrix<float>.Build.Dense(3, 2, new float[] { 1f, 1f, 0f, 0f, 0f, 1, });
            var predictions = Matrix<float>.Build.Dense(3, 2, new float[] { 0.9f, 0.9f, 0.1f, .1f, .1f, .9f });

            var actual = sut.Loss(targets, predictions);
            Assert.AreEqual(0.105360545, actual, 0.001);
        }
    }
}
