using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.ImpurityCalculators;

namespace SharpLearning.DecisionTrees.Test.ImpurityCalculators
{
    [TestClass]
    public class ChildImpuritiesTest
    {
        [TestMethod]
        public void ChildImpurities_Equal()
        {
            var sut = new ChildImpurities(0.23, 0.55);
            
            var equal = new ChildImpurities(0.23, 0.55);
            var notEqual1 = new ChildImpurities(0.19, 0.55);
            var notEqual2 = new ChildImpurities(0.23, 0.213);

            Assert.AreEqual(equal, sut);
            Assert.AreNotEqual(notEqual1, sut);
            Assert.AreNotEqual(notEqual2, sut);
        }
    }
}
