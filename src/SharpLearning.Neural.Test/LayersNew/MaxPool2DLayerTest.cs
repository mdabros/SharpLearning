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

            var input = Variable.Create(5, 2, 20, 20);
            var sut = new MaxPool2DLayer(2, 2);
            sut.Initialize(input, executor);

            GradientCheckTools.CheckLayer(sut, executor, input, 1e-4f, new Random(21));
        }
    }
}
