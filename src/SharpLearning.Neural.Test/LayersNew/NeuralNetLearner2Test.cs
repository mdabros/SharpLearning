using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.LayersNew;
using SharpLearning.Neural.Loss;

namespace SharpLearning.Neural.Test.LayersNew
{
    [TestClass]
    public class NeuralNetLearner2Test
    {
        [TestMethod]
        public void NeuralNetLearner2_Learn()
        {
            var observationSize = 500;
            var channels = 3;
            var height = 32;
            var width = 32;

            var numberOfTargets = 10;

            var random = new Random(32);
            var inputShape = new TensorShape(observationSize, channels, height, width);
            var observations = Tensor<double>.Build(inputShape.Dimensions);
            observations.Map(v => random.NextDouble());

            var targetsIn = Enumerable.Range(0, observationSize)
                .Select(v => (double)random.Next(numberOfTargets))
                .ToArray();

            var targets = NeuralNetTest.Encode(targetsIn);

            var sut = new NeuralNet2();

            sut.Add(new MaxPool2DLayer(2, 2));
            sut.Add(new DenseLayer(100));
            sut.Add(new BatchNormalizationLayer());
            sut.Add(new ActivationLayer(Neural.Activations.Activation.Relu));

            // output layer 
            sut.Add(new DenseLayer(numberOfTargets)); // corresponds to number of classes
            sut.Add(new BatchNormalizationLayer());
            sut.Add(new ActivationLayer(Neural.Activations.Activation.SoftMax));

            var loss = new LogLoss();

            // learn model
            var learner = new NeuralNetLearner2(sut, loss, 0.001, 10);
            var model = learner.Learn(observations, targets);

            // predict using the model
            var observation = Tensor<double>.Build(1, channels, height, width);
            var predictions = Tensor<double>.Build(targets.Dimensions.ToArray());

            for (int j = 0; j < observations.Dimensions[0]; j++)
            {
                observations.SliceCopy(j, 1, observation);
                var prediction = model.Predict(observation);
                predictions.SetSlice(j, prediction);
            }
            
            Trace.WriteLine($"Prediction Loss: {loss.Loss(targets, predictions)}");
        }
    }
}
