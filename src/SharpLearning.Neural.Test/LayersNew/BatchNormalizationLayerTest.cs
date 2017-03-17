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

            var input = new TensorShape(10, 1, 28, 28);
            var sut = new BatchNormalizationLayer(new BatchNormalization());
            sut.Initialize(input, executor);

            GradientCheckTools.CheckLayer(sut, executor, input, 1e-4f, new Random(21));
        }
    }
}
