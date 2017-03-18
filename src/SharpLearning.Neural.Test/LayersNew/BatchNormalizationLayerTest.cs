using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
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

            var input = Variable.Create(1, 3, 3, 3);
            var sut = new BatchNormalizationLayer();
            sut.Initialize(input, executor);

            GradientCheckTools.CheckLayer(sut, executor, input, 1e-4f, new Random(21));
        }
    }
}
