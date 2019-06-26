using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.TargetEncoders;

namespace SharpLearning.Neural.Test.TargetEncoders
{
    [TestClass]
    public class OneOfNTargetEncoderTest
    {
        [TestMethod]
        public void OneOfNTargetEncoder_Encode()
        {
            var targets = new double[] { 1, 1, 0, 0, 2, 0, 2 };
            var sut = new OneOfNTargetEncoder();

            var actual = sut.Encode(targets);
            var expected = Matrix<float>.Build.Dense(7, 3, new float[] { 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1 });
            
            Trace.WriteLine(expected.ToString());
            Assert.AreEqual(expected, actual);
        }
    }
}
