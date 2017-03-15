using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.LayersNew;
using SharpLearning.Neural.Providers.DotNetOp;

namespace SharpLearning.Neural.Test.LayersNew
{
    [TestClass]
    public class BatchNormalizationLayerTest
    {
        [TestMethod]
        public void BatchNormalizationLayer_GradientCheck()
        {
            var executor = new Executor();
            var sut = new BatchNormalizationLayer(new BatchNormalization());

            sut.Initialize(new TensorShape(10, 1, 28, 28), executor);

            sut.Forward(executor);
            sut.Backward(executor);
        }
    }
}
