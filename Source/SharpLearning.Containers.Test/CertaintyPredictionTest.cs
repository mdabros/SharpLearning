using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Containers.Test
{
    [TestClass]
    public class CertaintyPredictionTest
    {
        readonly CertaintyPrediction Sut = new CertaintyPrediction(1.0, 0.4);
        readonly CertaintyPrediction Equal = new CertaintyPrediction(1.0, 0.4);
        readonly CertaintyPrediction NotEqual1 = new CertaintyPrediction(0.0, 0.4);
        readonly CertaintyPrediction NotEqual2 = new CertaintyPrediction(1.0, 0.65);


        [TestMethod]
        public void CertaintyPrediction_Prediction_Equals()
        {
            Assert.AreEqual(Equal, Sut);
            Assert.AreNotEqual(NotEqual1, Sut);
            Assert.AreNotEqual(NotEqual2, Sut);
        }

        [TestMethod]
        public void CertaintyPrediction_Prediction_NotEqual_Operator()
        {
            Assert.IsTrue(NotEqual1 != Sut);
            Assert.IsTrue(NotEqual2 != Sut);
            Assert.IsFalse(Equal != Sut);
        }

        [TestMethod]
        public void CertaintyPrediction_Prediction_Equal_Operator()
        {
            Assert.IsFalse(NotEqual1 == Sut);
            Assert.IsFalse(NotEqual2 == Sut);
            Assert.IsTrue(Equal == Sut);
        }
    }
}
