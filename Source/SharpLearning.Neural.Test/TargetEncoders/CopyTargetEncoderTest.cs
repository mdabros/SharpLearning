using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.TargetEncoders;
using MathNet.Numerics.LinearAlgebra;

namespace SharpLearning.Neural.Test.TargetEncoders
{
    [TestClass]
    public class CopyTargetEncoderTest
    {
        [TestMethod]
        public void CopyTargetEncoder_Encode()
        {
            var targets = new double[] { 1, 1, 0, 0, 2, 0, 2 };
            var sut = new CopyTargetEncoder();

            var actual = sut.Encode(targets);
            var expected = Matrix<float>.Build.Dense(7, 1, new float[] { 1, 1, 0, 0, 2, 0, 2 });

            Assert.AreEqual(actual, expected);
        }
    }
}
