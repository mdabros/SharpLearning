using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.LeafValueFactories;
using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.Test.LeafValueFactories
{
    [TestClass]
    public class RegressionLeafValueFactoryTest
    {
        [TestMethod]
        public void RegressionLeafValueFactory_Calculate()
        {
            var values = new double[] { 1, 1, 1, 2, 2 }; 
            var sut = new RegressionLeafValueFactory();
            var actual = sut.Calculate(values);

            Assert.AreEqual(1.4, actual);
        }

        [TestMethod]
        public void RegressionLeafValueFactory_Calculate_Interval()
        {
            var values = new double[] { 1, 1, 1, 2, 2 };
            var sut = new RegressionLeafValueFactory();
            var actual = sut.Calculate(values, Interval1D.Create(1, 5));

            Assert.AreEqual(1.5, actual);
        }
    }
}
