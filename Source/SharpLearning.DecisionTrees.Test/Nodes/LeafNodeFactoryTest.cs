using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.Nodes;

namespace SharpLearning.DecisionTrees.Test.Nodes
{
    [TestClass]
    public class LeafNodeFactoryTest
    {
        [TestMethod]
        public void LeafNodeFactory_CreateProbabilityLeafNode()
        {
            var sut = new LeafNodeFactory();
            var actual = sut.Create(-1, 0.0, -1,
                new double[] { 1.0, 2.0 }, new double[] { 1.0, 2.0 });

            Assert.AreEqual(actual.GetType(), typeof(ProbabilityLeafNode));
        }

        [TestMethod]
        public void LeafNodeFactory_CreateLeafNode()
        {
            var sut = new LeafNodeFactory();
            var actual1 = sut.Create(-1, 0.0, -1,
                new double[0], new double[0]);
            var actual2 = sut.Create(-1, 0.0, -1,
                new double[0], new double[] { 1.0, 2.0 });
            var actual3 = sut.Create(-1, 0.0, -1,
                new double[]{ 1.0, 2.0 }, new double[0]);

            Assert.AreEqual(actual1.GetType(), typeof(LeafNode));
            Assert.AreEqual(actual2.GetType(), typeof(LeafNode));
            Assert.AreEqual(actual3.GetType(), typeof(LeafNode));
        }
    }
}
