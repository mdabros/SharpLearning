using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.Nodes;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Test.Nodes
{
    [TestClass]
    public class NodeExtensionsTest
    {
        [TestMethod]
        public void NodeExtensions_UpdateParent()
        {
            var nodes = new List<INode> { new SplitNode(-1, 2.0, -1, -1, 0) };
            var left = new LeafNode(1, 5.0, 1);
            var right = new LeafNode(1, 5.0, 2);

            nodes.UpdateParent(nodes[0], left, NodePositionType.Left);
            Assert.AreEqual(nodes[0].LeftIndex, left.NodeIndex);

            nodes.UpdateParent(nodes[0], right, NodePositionType.Right);
            Assert.AreEqual(nodes[0].LeftIndex, left.NodeIndex);
            Assert.AreEqual(nodes[0].RightIndex, right.NodeIndex);
        }
    }
}
