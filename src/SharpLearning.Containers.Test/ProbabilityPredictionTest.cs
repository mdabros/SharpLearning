using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Containers.Test
{
    [TestClass]
    public class ProbabilityPredictionTest
    {
        readonly ProbabilityPrediction Sut = new ProbabilityPrediction(1.0, new Dictionary<double, double> { { 1.0, .9 }, { 0.0, 0.3 } });
        readonly ProbabilityPrediction Equal = new ProbabilityPrediction(1.0, new Dictionary<double, double> { { 1.0, .9 }, { 0.0, 0.3 } });
        readonly ProbabilityPrediction NotEqual1 = new ProbabilityPrediction(0.0, new Dictionary<double, double> { { 1.0, .3 }, { 0.0, 0.8 } });
        readonly ProbabilityPrediction NotEqual2 = new ProbabilityPrediction(1.0, new Dictionary<double, double> { { 1.0, .78 }, { 0.0, 0.22 } });


        [TestMethod]
        public void ProbabilityPrediction_Prediction_Equals()
        {
            Assert.AreEqual(Equal, Sut);
            Assert.AreNotEqual(NotEqual1, Sut);
            Assert.AreNotEqual(NotEqual2, Sut);
        }

        [TestMethod]
        public void ProbabilityPrediction_Prediction_NotEqual_Operator()
        {
            Assert.IsTrue(NotEqual1 != Sut);
            Assert.IsTrue(NotEqual2 != Sut);
            Assert.IsFalse(Equal != Sut);
        }

        [TestMethod]
        public void ProbabilityPrediction_Prediction_Equal_Operator()
        {
            Assert.IsFalse(NotEqual1 == Sut);
            Assert.IsFalse(NotEqual2 == Sut);
            Assert.IsTrue(Equal == Sut);
        }
    }
}
