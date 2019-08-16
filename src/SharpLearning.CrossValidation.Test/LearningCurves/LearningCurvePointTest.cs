using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.LearningCurves;

namespace SharpLearning.CrossValidation.Test.LearningCurves
{
    [TestClass]
    public class LearningCurvePointTest
    {
        [TestMethod]
        public void BiasVarianceLearningCurvePoint_Equals()
        {
            var sut = new LearningCurvePoint(10, 1.0, 2.0);
            var equal = new LearningCurvePoint(10, 1.0, 2.0);

            var notEqual1 = new LearningCurvePoint(11, 1.0, 2.0);
            var notEqual2 = new LearningCurvePoint(10, 1.2, 2.0);
            var notEqual3 = new LearningCurvePoint(10, 1.0, 2.1);

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
