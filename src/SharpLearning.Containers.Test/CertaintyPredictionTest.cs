using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Containers.Test
{
    [TestClass]
    public class CertaintyPredictionTest
    {
        readonly CertaintyPrediction m_sut = new CertaintyPrediction(1.0, 0.4);
        readonly CertaintyPrediction m_equal = new CertaintyPrediction(1.0, 0.4);
        readonly CertaintyPrediction m_notEqual1 = new CertaintyPrediction(0.0, 0.4);
        readonly CertaintyPrediction m_notEqual2 = new CertaintyPrediction(1.0, 0.65);

        [TestMethod]
        public void CertaintyPrediction_Prediction_Equals()
        {
            Assert.AreEqual(m_equal, m_sut);
            Assert.AreNotEqual(m_notEqual1, m_sut);
            Assert.AreNotEqual(m_notEqual2, m_sut);
        }

        [TestMethod]
        public void CertaintyPrediction_Prediction_NotEqual_Operator()
        {
            Assert.IsTrue(m_notEqual1 != m_sut);
            Assert.IsTrue(m_notEqual2 != m_sut);
            Assert.IsFalse(m_equal != m_sut);
        }

        [TestMethod]
        public void CertaintyPrediction_Prediction_Equal_Operator()
        {
            Assert.IsFalse(m_notEqual1 == m_sut);
            Assert.IsFalse(m_notEqual2 == m_sut);
            Assert.IsTrue(m_equal == m_sut);
        }
    }
}
