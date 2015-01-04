using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.BiasVarianceAnalysis;

namespace SharpLearning.CrossValidation.Test.BiasVarianceAnalysis
{
    [TestClass]
    public class BiasVarianceLearningCurvePointTest
    {
        [TestMethod]
        public void BiasVarianceLearningCurvePoint_Equals()
        {
            var sut = new BiasVarianceLearningCurvePoint(10, 1.0, 2.0);
            var equal = new BiasVarianceLearningCurvePoint(10, 1.0, 2.0);

            var notEqual1 = new BiasVarianceLearningCurvePoint(11, 1.0, 2.0);
            var notEqual2 = new BiasVarianceLearningCurvePoint(10, 1.2, 2.0);
            var notEqual3 = new BiasVarianceLearningCurvePoint(10, 1.0, 2.1);

            Assert.IsTrue(sut.Equals(equal));
            Assert.IsTrue(sut == equal);
            Assert.IsFalse(sut != equal);

            Assert.IsFalse(sut.Equals(notEqual1));
            Assert.IsTrue(sut != notEqual1);
            Assert.IsFalse(sut == notEqual1);
            
            Assert.IsFalse(sut.Equals(notEqual2));
            Assert.IsTrue(sut != notEqual2);
            Assert.IsFalse(sut == notEqual2);

            Assert.IsFalse(sut.Equals(notEqual3));
            Assert.IsTrue(sut != notEqual3);
            Assert.IsFalse(sut == notEqual3);
        }
    }
}
