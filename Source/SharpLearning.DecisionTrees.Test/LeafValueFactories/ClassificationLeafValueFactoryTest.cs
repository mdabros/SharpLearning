using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.LeafValueFactories;
using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.Test.LeafValueFactories
{
    [TestClass]
    public class ClassificationLeafValueFactoryTest
    {
        [TestMethod]
        public void ClassificationLeafValueFactory_Create()
        {
            var values = new double[] { 1, 1, 1, 2, 2 };
            var sut = new ClassificationLeafValueFactory();
            var actual = sut.Calculate(values);

            Assert.AreEqual(1, actual);
        }

        [TestMethod]
        public void ClassificationLeafValueFactory_Create_Equal()
        {
            var values = new double[] { 1, 1, 2, 2 };
            var sut = new ClassificationLeafValueFactory();
            var actual = sut.Calculate(values);

            Assert.AreEqual(1, actual);
        }

        [TestMethod]
        public void ClassificationLeafValueFactory_Create_Interval()
        {
            var values = new double[] { 1, 1, 1, 2, 2 };
            var sut = new ClassificationLeafValueFactory();
            var actual = sut.Calculate(values, Interval1D.Create(2, 5));

            Assert.AreEqual(2, actual);
        }
    }
}
