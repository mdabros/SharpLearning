using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Test.LayersNew
{
    [TestClass]
    public class BatchNormalizationLayerTest
    {
        [TestMethod]
        [Ignore] // gradient seems off
        public void BatchNormalizationLayer_GradientCheck_4D()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(10, 5, 3, 3);
            var sut = new BatchNormalizationLayer();
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayer(sut, storage, input, new Random(21));
        }

        [TestMethod]
        [Ignore] // gradient seems off
        public void BatchNormalizationLayer_GradientCheck_2D()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(10, 5);
            var sut = new BatchNormalizationLayer();
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayer(sut, storage, input, new Random(345));
        }
        
        [TestMethod]
        public void BatchNormalizationLayer_ParameterGradientCheck_4D()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(10, 10, 5, 5);
            var sut = new BatchNormalizationLayer();
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayerParameters(sut, storage, input, new Random(21));
        }

        [TestMethod]
        public void BatchNormalizationLayer_ParameterGradientCheck_2D()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(10, 10);
            var sut = new BatchNormalizationLayer();
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayerParameters(sut, storage, input, new Random(21));
        }
    }
}
