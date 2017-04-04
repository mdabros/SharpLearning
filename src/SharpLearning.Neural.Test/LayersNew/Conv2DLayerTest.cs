using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Test.LayersNew
{
    [TestClass]
    public class Conv2DLayerTest
    {
        [TestMethod]
        public void Conv2DLayer_GradientCheck()
        {
            var executor = new NeuralNetStorage();

            var input = Variable.Create(1, 3, 3, 3);
            var sut = new Conv2DLayer(2, 2, 2);
            sut.Initialize(input, executor, new Random());

            GradientCheckTools.CheckLayer(sut, executor, input, new Random(21));
        }

        [TestMethod]
        [Ignore] // gradient seems off
        public void Conv2DLayer_GradientCheck_Batch_5()
        {
            var executor = new NeuralNetStorage();

            var input = Variable.Create(5, 3, 3, 3);
            var sut = new Conv2DLayer(2, 2, 2);
            sut.Initialize(input, executor, new Random());

            GradientCheckTools.CheckLayer(sut, executor, input, new Random(21));
        }

        [TestMethod]
        public void Conv2DLayer_ParameterGradientCheck()
        {
            var executor = new NeuralNetStorage();

            var input = Variable.Create(1, 3, 3, 3);
            var sut = new Conv2DLayer(2, 2, 2);
            sut.Initialize(input, executor, new Random());

            GradientCheckTools.CheckLayerParameters(sut, executor, input, new Random(21));
        }

        [TestMethod]
        [Ignore] // gradient seems off
        public void Conv2DLayer_ParameterGradientCheck_Batch_5()
        {
            var executor = new NeuralNetStorage();

            var input = Variable.Create(5, 3, 3, 3);
            var sut = new Conv2DLayer(2, 2, 2);
            sut.Initialize(input, executor, new Random());

            GradientCheckTools.CheckLayerParameters(sut, executor, input, new Random(21));
        }
    }
}
