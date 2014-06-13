using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.LeafFactories;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Nodes;

namespace SharpLearning.DecisionTrees.Test.LeafFactories
{
    [TestClass]
    public class ClassificationLeafFactoryTest
    {
        [TestMethod]
        public void ClassificationLeafValueFactory_Create()
        {
            var values = new double[] { 1, 1, 1, 2, 2 };
            var sut = new ClassificationLeafFactory();
            var actual = sut.Create(new ContinousBinaryDecisionNode(), values);

            Assert.AreEqual(1, actual.Value);
            Assert.AreEqual(-1, actual.FeatureIndex);
            Assert.IsTrue(actual.IsLeaf());
            Assert.IsNotNull(actual.Parent);
        }

        [TestMethod]
        public void ClassificationLeafValueFactory_Create_Equal()
        {
            var values = new double[] { 1, 1, 2, 2 };
            var sut = new ClassificationLeafFactory();
            var actual = sut.Create(new ContinousBinaryDecisionNode(), values);

            Assert.AreEqual(1, actual.Value);
            Assert.AreEqual(-1, actual.FeatureIndex);
            Assert.IsTrue(actual.IsLeaf());
            Assert.IsNotNull(actual.Parent);
        }

        [TestMethod]
        public void ClassificationLeafValueFactory_Create_Interval()
        {
            var values = new double[] { 1, 1, 1, 2, 2 };
            var sut = new ClassificationLeafFactory();
            var actual = sut.Create(new ContinousBinaryDecisionNode(), values, Interval1D.Create(2, 5));

            Assert.AreEqual(2, actual.Value);
            Assert.AreEqual(-1, actual.FeatureIndex);
            Assert.IsTrue(actual.IsLeaf());
            Assert.IsNotNull(actual.Parent);
        }
    }
}
