using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.LayersNew;
using SharpLearning.Neural.Providers.DotNetOp;

namespace SharpLearning.Neural.Test.LayersNew
{
    [TestClass]
    public class DenseLayerTest
    {
        [TestMethod]
        public void DenseLayer_GradientCheck()
        {
            var executor = new Executor();

            var input = Variable.Create(1, 5, 1, 1);
            var sut = new DenseLayer(3);
            sut.Initialize(input, executor, new Random());

            GradientCheckTools.CheckLayer(sut, executor, input, 1e-4f, new Random(21));
        }

        [TestMethod]
        public void DenseLayer_ParameterGradientCheck()
        {
            var executor = new Executor();

            var input = Variable.Create(10, 5, 1, 1);
            var sut = new DenseLayer(3);
            sut.Initialize(input, executor, new Random());

            GradientCheckTools.CheckLayerParameters(sut, executor, input, 1e-4f, new Random(21));
        }
    }
}
