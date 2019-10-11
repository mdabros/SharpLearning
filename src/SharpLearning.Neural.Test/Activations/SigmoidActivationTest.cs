using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.Activations;

namespace SharpLearning.Neural.Test.Activations
{
    [TestClass]
    public class SigmoidActivationTest
    {

        [TestMethod]
        public void SigmoidActivation_Activiation()
        {
            var actual = new float[] { -20, -10, -5, -1, 0, 1, 5, 10, 20 };
            var sut = new SigmoidActivation();
            sut.Activation(actual);

            var expected = new float[] { 2.06115369E-09f, 4.539787E-05f, 0.006692851f,
               0.268941432f, 0.5f,0.7310586f,0.9933072f,0.9999546f, 1 };

            Assert.AreEqual(expected[0], actual[0]);
            Assert.AreEqual(expected[1], actual[1]);
            Assert.AreEqual(expected[2], actual[2]);
            Assert.AreEqual(expected[3], actual[3]);
            Assert.AreEqual(expected[4], actual[4]);
            Assert.AreEqual(expected[5], actual[5]);
            Assert.AreEqual(expected[6], actual[6]);
            Assert.AreEqual(expected[7], actual[7]);
            Assert.AreEqual(expected[8], actual[8]);
        }

        [TestMethod]
        public void SigmoidActivation_Derivative()
        {
            var activatedSigmoid = new float[] { 2.06115369E-09f, 4.539787E-05f, 0.006692851f,
               0.268941432f, 0.5f,0.7310586f,0.9933072f,0.9999546f, 1 };

            var actual = new float[9];
            var sut = new SigmoidActivation();

            sut.Derivative(activatedSigmoid, actual);

            var expected = new float[] { 2.06115369E-09f, 4.53958055E-05f,
                0.00664805667f, 0.196611941f, 0.25f, 0.196611926f, 0.006648033f,
                4.54166766E-05f, 1 };

            Assert.AreEqual(expected[0], actual[0]);
            Assert.AreEqual(expected[1], actual[1]);
            Assert.AreEqual(expected[2], actual[2]);
            Assert.AreEqual(expected[3], actual[3]);
            Assert.AreEqual(expected[4], actual[4]);
            Assert.AreEqual(expected[5], actual[5]);
            Assert.AreEqual(expected[6], actual[6]);
            Assert.AreEqual(expected[7], actual[7]);
            Assert.AreEqual(expected[8], actual[8]);
        }
    }
}
