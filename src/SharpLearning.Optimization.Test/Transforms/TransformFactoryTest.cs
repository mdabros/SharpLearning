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
            var log10 = TransformFactory.Create(Transform.Log10);
            Assert.AreEqual(typeof(Log10Transform), log10.GetType());
            var exponentialAverage = TransformFactory.Create(Transform.ExponentialAverage);
            Assert.AreEqual(typeof(ExponentialAverageTransform), exponentialAverage.GetType());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TransformFactory_Create_Throws_On_Invalid_Transform()
        {
            TransformFactory.Create((Transform)3);
        }
    }
}
