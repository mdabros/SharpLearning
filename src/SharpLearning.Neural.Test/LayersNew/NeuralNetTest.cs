using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.LayersNew;
using SharpLearning.Neural.Loss;

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
            var observations = Tensor<double>.Build(inputShape.Dimensions);
            observations.Map(v => random.NextDouble());

            var targets = Tensor<double>.Build(10, 2);
            targets.Map(v => random.Next(2));

            var sut = new NeuralNet2();

            sut.Add(new DenseLayer(100));
            sut.Add(new BatchNormalizationLayer());
            sut.Add(new ActivationLayer(Neural.Activations.Activation.Relu));

            // output layer 
            sut.Add(new DenseLayer(2)); // corresponds to number of classes
            sut.Add(new BatchNormalizationLayer());
            sut.Add(new ActivationLayer(Neural.Activations.Activation.SoftMax));

            // hack because of missing input layer.
            sut.Initialize(new Variable(inputShape), random);
            
            sut.SetNextBatch(observations, targets);

            sut.Forward();
            sut.Backward();

            var predictions = sut.BatchPredictions();
            var lossFunc = new LogLoss();
            var loss = lossFunc.Loss(targets, predictions);
            Trace.WriteLine($"Loss: {loss}");
        }
    }
}
