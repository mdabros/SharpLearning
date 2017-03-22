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
            var batchSize = 100;
            var numberOfTargets = 10;

            var random = new Random(32);
            var inputShape = new TensorShape(batchSize, 1, 25, 25);
            var observations = Tensor<double>.Build(inputShape.Dimensions);
            observations.Map(v => random.NextDouble());

            var targetsIn = Enumerable.Range(0, batchSize)
                .Select(v => (double)random.Next(numberOfTargets))
                .ToArray();

            var targets = Encode(targetsIn);

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
            sut.Initialize(new Variable(inputShape), random);
            
            var parameters = new List<Data<double>>();
            sut.GetTrainableParameters(parameters);

            var lossFunc = new LogLoss();
            var iteration = 10;

            for (int i = 0; i < iteration; i++)
            {
                sut.SetNextBatch(observations, targets);

                sut.Forward();
                sut.Backward();
                
                Trace.WriteLine($"Loss: {lossFunc.Loss(targets, sut.BatchPredictions())}");

                optimizer.UpdateParameters(parameters);
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
