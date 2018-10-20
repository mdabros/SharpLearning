using System;
using System.Collections.Generic;
using CNTK;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Test
{
    [TestClass]
    public class LearnersTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void SGD_Parameters_Is_Null_Throws()
        {
            Learners.SGD(null);
        }
        
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void SGD_LearningRate_Below_Zero_Throws()
        {
            Learners.SGD(new List<Parameter>(), learningRate: -0.1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void SGD_L1Regularization_Below_Zero_Throws()
        {
            Learners.SGD(new List<Parameter>(), l1Regularization: -0.1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void SGD_L2Regularization_Below_Zero_Throws()
        {
            Learners.SGD(new List<Parameter>(), l2Regularization: -0.1);
        }

        [TestMethod]
        public void SetAdditionalOptions()
        {
            var expectedL1Regularization = 0.1;
            var expectedL2Regularization = 0.001;

            var sut = Learners.SetAdditionalOptions(expectedL1Regularization, expectedL2Regularization);

            Assert.AreEqual(expectedL1Regularization, sut.l1RegularizationWeight);
            Assert.AreEqual(expectedL2Regularization, sut.l2RegularizationWeight);
        }
    }
}
