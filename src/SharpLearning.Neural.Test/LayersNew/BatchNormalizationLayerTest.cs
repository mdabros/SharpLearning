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
        [Ignore] // gradient seems off
        public void BatchNormalizationLayer_GradientCheck()
        {
            var executor = new Executor();

            var input = Variable.Create(10, 5, 1, 1);
            var sut = new BatchNormalizationLayer();
            sut.Initialize(input, executor, new Random());

            GradientCheckTools.CheckLayer(sut, executor, input, 1e-4f, new Random(21));
        }

        [TestMethod]
        [Ignore] // gradient seems off
        public void BatchNormalizationLayer_ParameterGradientCheck()
        {
            var executor = new Executor();

            var input = Variable.Create(10, 5, 1, 1);
            var sut = new BatchNormalizationLayer();
            sut.Initialize(input, executor, new Random());

            GradientCheckTools.CheckLayerParameters(sut, executor, input, 1e-4f, new Random(21));
        }
    }
}
