using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.Nodes;

namespace SharpLearning.DecisionTrees.Test.Nodes
{
    [TestClass]
    public class BinaryDecisionNodeExtensionsTest
    {
        [TestMethod]
        public void BinaryDecisionNodeExtensions_AddChild_Left()
        {
            var parent = new ContinousBinaryDecisionNode();
            var left = new ContinousBinaryDecisionNode();

            parent.AddChild(NodePositionType.Left, left);
            
            Assert.IsTrue(parent.IsRoot());
            Assert.IsTrue(parent.Right == null);
            Assert.IsTrue(parent.Left != null);
            Assert.IsTrue(left.IsLeaf());
            Assert.IsFalse(left.IsRoot());
        }

        [TestMethod]
        public void BinaryDecisionNodeExtensions_AddChild_Right()
        {
            var parent = new ContinousBinaryDecisionNode();
            var right = new ContinousBinaryDecisionNode();

            parent.AddChild(NodePositionType.Right, right);

            Assert.IsTrue(parent.IsRoot());
            Assert.IsTrue(parent.Right != null);
            Assert.IsTrue(parent.Left == null);
            Assert.IsTrue(right.IsLeaf());
            Assert.IsFalse(right.IsRoot());
        }

        [TestMethod]
        public void BinaryDecisionNodeExtensions_AddChild_Root()
        {
            var parent = new ContinousBinaryDecisionNode();
            parent.AddChild(NodePositionType.Root, parent);

            Assert.IsTrue(parent.IsRoot());
            Assert.IsTrue(parent.Right == null);
            Assert.IsTrue(parent.Left == null);
        }
    }
}
