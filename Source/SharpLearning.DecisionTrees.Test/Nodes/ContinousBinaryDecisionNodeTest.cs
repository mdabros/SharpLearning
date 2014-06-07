using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.Nodes;

namespace SharpLearning.DecisionTrees.Test.Nodes
{
    [TestClass]
    public class ContinousBinaryDecisionNodeTest
    {
        [TestMethod]
        public void ContinousBinaryDecisionNode_IsRoot()
        {
            var parent = new ContinousBinaryDecisionNode();
            var left = new ContinousBinaryDecisionNode();
            var right = new ContinousBinaryDecisionNode();

            parent.Left = left;
            parent.Right = right;

            Assert.IsTrue(parent.IsRoot());

            Assert.IsFalse(left.IsRoot());
            Assert.IsTrue(left.IsLeaf());
            Assert.IsFalse(right.IsRoot());
            Assert.IsTrue(right.IsLeaf());
        }

        [TestMethod]
        public void ContinousBinaryDecisionNode_Properties()
        {
            var sut = new ContinousBinaryDecisionNode();
            sut.FeatureIndex = 1;
            sut.Value = 30;

            Assert.AreEqual(1, sut.FeatureIndex);
            Assert.AreEqual(30, sut.Value);
        }

        [TestMethod]
        public void ContinousBinaryDecisionNode_Predict()
        {
            var sut = new ContinousBinaryDecisionNode { FeatureIndex = 0, Value = 20 };
            var left = new ContinousBinaryDecisionNode { FeatureIndex = -1, Value = 30 };
            var right = new ContinousBinaryDecisionNode { FeatureIndex = -1, Value = 10 };
            sut.Left = left;
            sut.Right = right;
    
            var observation1 = new double[] { 25 };
            Assert.AreEqual(30, sut.Predict(observation1));

            var observation2 = new double[] { 15 };
            Assert.AreEqual(10, sut.Predict(observation2));
        }
    }
}
