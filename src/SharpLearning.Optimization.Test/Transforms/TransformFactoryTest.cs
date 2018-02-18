using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Optimization.Transforms;

namespace SharpLearning.Optimization.Test.Transforms
{
    [TestClass]
    public class TransformFactoryTest
    {
        [TestMethod]
        public void TransformFactory_Create()
        {
            var linear = TransformFactory.Create(Transform.Linear);
            Assert.AreEqual(typeof(LinearTransform), linear.GetType());
            var logarithmic = TransformFactory.Create(Transform.Linear);
            Assert.AreEqual(typeof(LinearTransform), logarithmic.GetType());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TransformFactory_Create_Throws_On_Invalid_Transform()
        {
            TransformFactory.Create((Transform)2);
        }
    }
}
