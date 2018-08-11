using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Accord.DataSets;
using CntkExtensions;
using CntkExtensions.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Examples
{
    /// <summary>
    /// Example from Chapter 02: First look at a Neural Network:
    /// https://github.com/mdabros/deep-learning-with-python-notebooks/blob/master/2.1-a-first-look-at-a-neural-network.ipynb
    /// </summary>
    [TestClass]
    public class Ch_02_First_Look_At_A_Neural_Network
    {
        enum MnistDataSplit { Train, Test };

        [TestMethod]
        public void Run()
        {
            var inputShape = new int[] { 28, 28, 1 };
            var numberOfClasses = 10;
            var outputShape = new int[] { numberOfClasses };

            (var trainImages, var trainTargets) = LoadMnistData(inputShape, outputShape, MnistDataSplit.Train);
            (var testImages, var testTargets) = LoadMnistData(inputShape, outputShape, MnistDataSplit.Test);

            var network = new Sequential(Layers.Input(inputShape));

            network.Add(x => Layers.Dense(x, units: 512));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: numberOfClasses));
            network.Add(x => Layers.Softmax(x));

            network.Compile(p => Learners.RMSProp(p),
               (t, p) => Losses.CategoricalCrossEntropy(t, p),
               (t, p) => Metrics.Accuracy(t, p));

            network.Fit(trainImages, trainTargets, epochs: 25, batchSize: 128);

            (var loss, var metric) = network.Evaluate(testImages, testTargets);

            Trace.WriteLine($"Test set - Loss: {loss}, Metric: {metric}");
        }

        static (Tensor observations, Tensor targets) LoadMnistData(int[] inputShape, int[] outputShape, MnistDataSplit dataSplit)
        {
            var mnist = new MNIST(Directory.GetCurrentDirectory());
            var dataSet = dataSplit == MnistDataSplit.Train ? mnist.Training : mnist.Testing;

            var observationCount = dataSet.Item2.Length;
            var dataSize = inputShape.Aggregate((d1, d2) => d1 * d2);

            // flatten data.
            var observationsData = dataSet.Item1
                .Select(s => s.ToDense(dataSize)) // set fixed dataSize.
                .SelectMany(d => d)                
                .Select(d => (float) d / 255) // transform pixel values to be between 0 and 1.
                .ToArray();

            var targetsData = dataSet.Item2
                .Select(d => (float)d)
                .ToArray();

            var observationsShape = new List<int>(inputShape);
            observationsShape.Add(observationCount);
            var observations = new Tensor(observationsData, observationsShape.ToArray());

            var oneHotTargetsData = targetsData.EncodeOneHot();

            var targetsShape = new List<int>(outputShape);
            targetsShape.Add(observationCount);
            var targets = new Tensor(oneHotTargetsData, targetsShape.ToArray());

            return (observations, targets);
        }
    }
}
