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
            var nodes = new List<Node> { new Node(-1, 2.0, -1, -1, 0, -1) };
            var left = new Node(1, 5.0, -1, -1, 1, -1);
            var right = new Node(1, 5.0, -1, -1, 2, -1);

            nodes.UpdateParent(nodes[0], left, NodePositionType.Left);
            Assert.AreEqual(nodes[0].LeftIndex, left.NodeIndex);

            nodes.UpdateParent(nodes[0], right, NodePositionType.Right);
            Assert.AreEqual(nodes[0].LeftIndex, left.NodeIndex);
            Assert.AreEqual(nodes[0].RightIndex, right.NodeIndex);
        }
    }
}
