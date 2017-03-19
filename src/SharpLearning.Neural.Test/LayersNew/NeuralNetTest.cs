using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Test.LayersNew
{
    [TestClass]
    public class NeuralNetTest
    {
        [TestMethod]
        public void NeuralNet_Test()
        {
            var random = new Random(32);
            var inputShape = new TensorShape(10, 25, 25);
            var observations = Tensor<float>.Build(inputShape.Dimensions);
            observations.Map(v => (float)random.NextDouble());

            var targets = Tensor<float>.Build(10, 1);

            var sut = new NeuralNet2();
            sut.Add(new DenseLayer(30));
            sut.Add(new ActivationLayer(Neural.Activations.Activation.Relu));
            sut.Add(new DenseLayer(1)); // corresponds to number of classes
            sut.Add(new ActivationLayer(Neural.Activations.Activation.SoftMax));

            // hack because of missing input layer.
            sut.Initialize(new Variable(inputShape), random);
            
            sut.SetNextBatch(observations, targets);
            sut.Forward();
            sut.Backward();
        }
    }
}
