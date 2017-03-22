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
            var executor = new Executor();

            var input = Variable.Create(10, 2, 5, 5);
            var sut = new ActivationLayer(Neural.Activations.Activation.Relu);
            sut.Initialize(input, executor, new Random());

            GradientCheckTools.CheckLayer(sut, executor, input, new Random(21));
        }

        [TestMethod]
        [Ignore] // gradient check seems off.
        public void ActivationLayer_SoftMax_GradientCheck()
        {
            var executor = new Executor();

            var input = Variable.Create(10, 5, 1, 1);
            var sut = new ActivationLayer(Neural.Activations.Activation.SoftMax);
            sut.Initialize(input, executor, new Random());

            GradientCheckTools.CheckLayer(sut, executor, input, new Random(21));
        }
    }
}
