using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.LeafFactories;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Nodes;

namespace SharpLearning.DecisionTrees.Test.LeafFactories
{
    [TestClass]
    public class RegressionLeafFactoryTest
    {
        [TestMethod]
        public void RegressionLeafValueFactory_Calculate()
        {
            var values = new double[] { 1, 1, 1, 2, 2 }; 
            var sut = new RegressionLeafFactory();
            var actual = sut.Create(new ContinousBinaryDecisionNode(), values);

            Assert.AreEqual(1.4, actual.Value);
            Assert.AreEqual(-1, actual.FeatureIndex);
            Assert.IsTrue(actual.IsLeaf());
            Assert.IsNotNull(actual.Parent);
        }

        [TestMethod]
        public void RegressionLeafValueFactory_Calculate_Interval()
        {
            var values = new double[] { 1, 1, 1, 2, 2 };
            var sut = new RegressionLeafFactory();
            var actual = sut.Create(new ContinousBinaryDecisionNode(), values, Interval1D.Create(1, 5));

            Assert.AreEqual(1.5, actual.Value);
            Assert.AreEqual(-1, actual.FeatureIndex);
            Assert.IsTrue(actual.IsLeaf());
            Assert.IsNotNull(actual.Parent);
        }
    }
}
