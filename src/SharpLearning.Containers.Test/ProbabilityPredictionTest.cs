using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Containers.Test
{
    [TestClass]
    public class ProbabilityPredictionTest
    {
        readonly ProbabilityPrediction m_sut = new(1.0, new Dictionary<double, double> { { 1.0, .9 }, { 0.0, 0.3 } });
        readonly ProbabilityPrediction m_equal = new(1.0, new Dictionary<double, double> { { 1.0, .9 }, { 0.0, 0.3 } });
        readonly ProbabilityPrediction m_notEqual1 = new(0.0, new Dictionary<double, double> { { 1.0, .3 }, { 0.0, 0.8 } });
        readonly ProbabilityPrediction m_notEqual2 = new(1.0, new Dictionary<double, double> { { 1.0, .78 }, { 0.0, 0.22 } });


        [TestMethod]
        public void ProbabilityPrediction_Prediction_Equals()
        {
            Assert.AreEqual(m_equal, m_sut);
            Assert.AreNotEqual(m_notEqual1, m_sut);
            Assert.AreNotEqual(m_notEqual2, m_sut);
        }

        [TestMethod]
        public void ProbabilityPrediction_Prediction_NotEqual_Operator()
        {
            Assert.IsTrue(m_notEqual1 != m_sut);
            Assert.IsTrue(m_notEqual2 != m_sut);
            Assert.IsFalse(m_equal != m_sut);
        }

        [TestMethod]
        public void ProbabilityPrediction_Prediction_Equal_Operator()
        {
            Assert.IsFalse(m_notEqual1 == m_sut);
            Assert.IsFalse(m_notEqual2 == m_sut);
            Assert.IsTrue(m_equal == m_sut);
        }
    }
}
