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
            var executor = new NeuralNetStorage();

            var input = Variable.Create(1, 2, 5, 5);
            var sut = new DenseLayer(10);
            sut.Initialize(input, executor, new Random());

            GradientCheckTools.CheckLayer(sut, executor, input, new Random(21));
        }

        [TestMethod]
        public void DenseLayer_ParameterGradientCheck()
        {
            var executor = new NeuralNetStorage();

            var input = Variable.Create(10, 2, 5, 5);
            var sut = new DenseLayer(10);
            sut.Initialize(input, executor, new Random());

            GradientCheckTools.CheckLayerParameters(sut, executor, input, new Random(21));
        }
    }
}
