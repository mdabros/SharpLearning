using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Test.LayersNew
{
    [TestClass]
    public class MaxPool2DLayerTest
    {
        [TestMethod]
        public void MaxPool2DLayer_GradientCheck()
        {
            var executor = new Executor();

            var input = Variable.Create(10, 3, 5, 5);
            var sut = new MaxPool2DLayer(2, 2);
            sut.Initialize(input, executor, new Random());

            GradientCheckTools.CheckLayer(sut, executor, input, new Random(21));
        }
    }
}
