using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.SplitSearchers;

namespace SharpLearning.DecisionTrees.Test.SplitSearchers
{
    [TestClass]
    public class SplitResultTest
    {
        [TestMethod]
        public void SplitResult_Equals()
        {
            var sut = new SplitResult(1, 200, 13.4543, 20, 30);

            var equal = new SplitResult(1, 200, 13.4543, 20, 30);
            var notEqual1 = new SplitResult(2, 200, 13.4543, 20, 30);
            var notEqual2 = new SplitResult(1, 201, 13.4543, 20, 30);
            var notEqual3 = new SplitResult(1, 200, 16.4543, 20, 30);
            var notEqual4 = new SplitResult(1, 200, 13.4543, 10, 30);
            var notEqual5 = new SplitResult(1, 200, 13.4543, 20, 10);

            Assert.AreEqual(equal, sut);
            Assert.AreNotEqual(notEqual1, sut);
            Assert.AreNotEqual(notEqual2, sut);
            Assert.AreNotEqual(notEqual3, sut);
            Assert.AreNotEqual(notEqual4, sut);
            Assert.AreNotEqual(notEqual5, sut);
        }
    }
}
