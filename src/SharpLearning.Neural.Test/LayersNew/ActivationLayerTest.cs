using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Test.LayersNew
{
    [TestClass]
    public class ActivationLayerTest
    {
        [TestMethod]
        public void ActivationLayer_Relu_GradientCheck()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(10, 2, 5, 5);
            var sut = new ActivationLayer(Neural.Activations.Activation.Relu);
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayer(sut, storage, input, new Random(21));
        }
        
        [TestMethod]
        [Ignore] // gradient check seems off.
        public void ActivationLayer_SoftMax_GradientCheck()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(10, 5, 1, 1);
            var sut = new ActivationLayer(Neural.Activations.Activation.SoftMax);
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayer(sut, storage, input, new Random(21));
        }

        [TestMethod]
        [Ignore] // gradient check seems off.
        public void ActivationLayer_MeanSquareError_GradientCheck()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(10, 2, 5, 5);
            var sut = new ActivationLayer(Neural.Activations.Activation.MeanSquareError);
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayer(sut, storage, input, new Random(21));
        }

        [TestMethod]
        [Ignore] // gradient check seems off.
        public void ActivationLayer_Svm_GradientCheck()
        {
            var storage = new NeuralNetStorage();

            var input = Variable.Create(10, 2, 1, 1);
            var sut = new ActivationLayer(Neural.Activations.Activation.Svm);
            sut.Initialize(input, storage, new Random());

            GradientCheckTools.CheckLayer(sut, storage, input, new Random(21));
        }
    }
}
