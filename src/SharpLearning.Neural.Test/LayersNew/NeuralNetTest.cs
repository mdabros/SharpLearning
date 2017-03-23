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
    public class NeuralNetTest
    {
        [TestMethod]
        public void NeuralNet_Test()
        {
            var observationSize = 500;
            var channels = 3;
            var height = 32;
            var width = 32;

            var batchSize = 128;
            var numberOfTargets = 10;

            var random = new Random(32);
            var inputShape = new TensorShape(observationSize, channels, height, width);
            var observations = Tensor<double>.Build(inputShape.Dimensions);
            observations.Map(v => random.NextDouble());

            var targetsIn = Enumerable.Range(0, observationSize)
                .Select(v => (double)random.Next(numberOfTargets))
                .ToArray();

            var targets = Encode(targetsIn);

            var batchObservations = Tensor<double>.Build(batchSize, channels, height, width);
            var batchTargets = Tensor<double>.Build(batchSize, numberOfTargets);

            var optimizer = new NeuralNetOptimizer2(0.001, batchSize);
            var sut = new NeuralNet2();

            sut.Add(new MaxPool2DLayer(2, 2));
            sut.Add(new DenseLayer(100));
            sut.Add(new BatchNormalizationLayer());
            sut.Add(new ActivationLayer(Neural.Activations.Activation.Relu));

            // output layer 
            sut.Add(new DenseLayer(numberOfTargets)); // corresponds to number of classes
            sut.Add(new BatchNormalizationLayer());
            sut.Add(new ActivationLayer(Neural.Activations.Activation.SoftMax));

            // hack because of missing input layer.
            sut.Initialize(new Variable(batchObservations.Shape), random);
            
            var parameters = new List<Data<double>>();
            sut.GetTrainableParameters(parameters);

            var lossFunc = new LogLoss();
            var epochs = 10;

            var batcher = new Batcher();
            batcher.Initialize(observations.Shape, random.Next());

            for (int i = 0; i < epochs; i++)
            {
                batcher.Shuffle();
                var accumulatedLoss = 0.0;

                while (batcher.Next(batchSize, 
                    sut, observations, targets, 
                    batchObservations, batchTargets))
                {
                    sut.Forward();
                    sut.Backward();

                    optimizer.UpdateParameters(parameters);

                    var batchLoss = lossFunc.Loss(batchTargets, sut.BatchPredictions());
                    accumulatedLoss += batchLoss * batchSize;
                }

                var loss = accumulatedLoss / (double)observationSize;
                Trace.WriteLine($"Loss: {loss}");
            }
        }

        /// <summary>
        /// Encodes targets in a one-of-n structure. Target vector of with two classes [0, 1, 1, 0] becomes a matrix:
        /// 1 0
        /// 0 1
        /// 0 1
        /// 1 0
        /// Primary use is for classification
        /// </summary>
        /// <param name="targets"></param>
        /// <returns></returns>
        public Tensor<double> Encode(double[] targets)
        {
            var index = 0;
            var targetNameToTargetIndex = targets.Distinct().OrderBy(v => v)
                .ToDictionary(v => v, v => index++);

            var oneOfN = Tensor<double>.Build(targets.Length, targetNameToTargetIndex.Count);

            for (int i = 0; i < targets.Length; i++)
            {
                var target = targets[i];
                var targetIndex = i * targetNameToTargetIndex.Count + targetNameToTargetIndex[target];

                oneOfN.Data[targetIndex] = 1;
            }

            return oneOfN;
        }

        public Tensor<double> EncodeRegression(double[] targets)
        {
            return Tensor<double>.Build(targets, targets.Length);
        }
    }
}
