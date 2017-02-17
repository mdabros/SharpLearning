using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.FeatureTransformations.Normalization;

namespace SharpLearning.FeatureTransformations.Test.Normalization
{
    [TestClass]
    public class LinearNormalizerTest
    {
        [TestMethod]
        public void LinearNormalizer_Normalize()
        {
            var oldMin = -213.0;
            var oldMax = 2345.0;

            var newMin = -1.0;
            var newMax = 1.0;

            var sut = new LinearNormalizer();

            Assert.AreEqual(1.0, sut.Normalize(newMin, newMax, oldMin, oldMax, oldMax), 0.000001);
            Assert.AreEqual(-1.0, sut.Normalize(newMin, newMax, oldMin, oldMax, oldMin), 0.000001);

            Assert.AreEqual(-0.833463643471462, sut.Normalize(newMin, newMax, oldMin, oldMax, 0.0), 0.000001);
            Assert.AreEqual(0.730258014073495, sut.Normalize(newMin, newMax, oldMin, oldMax, 2000.0), 0.000001);
        }
    }
}
