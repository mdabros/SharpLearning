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
            var storage = new NeuralNetStorage();

            var input = Variable.Create(1, 2, 5, 5);
            var sut = new DenseLayer(10);
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayer(sut, storage, input, new Random(21));
        }

        [TestMethod]
        public void DenseLayer_ParameterGradientCheck()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(10, 2, 5, 5);
            var sut = new DenseLayer(10);
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayerParameters(sut, storage, input, new Random(21));
        }
    }
}
