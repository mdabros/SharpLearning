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
        [TestMethod]
        public void Run()
        {
            // Define the input and output shape.
            var inputShape = new int[] { 28, 28, 1 };
            var numberOfClasses = 10;
            var outputShape = new int[] { numberOfClasses };

            // Load train and test sets. 
            // Network will be trained using the training set,
            // and tested using the test set.
            (var trainImages, var trainTargets) = LoadMnistData(inputShape, outputShape, DataSplit.Train);
            (var testImages, var testTargets) = LoadMnistData(inputShape, outputShape, DataSplit.Test);

            // Create the network, and define the input shape.
            var network = new Sequential(Layers.Input(inputShape));

            // Add layes to the network.
            network.Add(x => Layers.Dense(x, units: 512));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: numberOfClasses));
            network.Add(x => Layers.Softmax(x));

            // Compile the network with the selected learner, loss and metric.
            network.Compile(p => Learners.RMSProp(p),
               (t, p) => Losses.CategoricalCrossEntropy(t, p),
               (t, p) => Metrics.Accuracy(t, p));

            // Train the model using the training set.
            network.Fit(trainImages, trainTargets, epochs: 5, batchSize: 128);

            // Evaluate the model using the test set.
            (var loss, var metric) = network.Evaluate(testImages, testTargets);

            // Write the test set loss and metric to debug output.
            Trace.WriteLine($"Test set - Loss: {loss}, Metric: {metric}");
        }

        static (Tensor observations, Tensor targets) LoadMnistData(int[] inputShape, int[] outputShape, DataSplit dataSplit)
        {
            // Load mnist data set using Accord.DataSets.
            var mnist = new MNIST(Directory.GetCurrentDirectory());
            var dataSet = dataSplit == DataSplit.Train ? mnist.Training : mnist.Testing;

            var observationCount = dataSet.Item2.Length;
            var dataSize = inputShape.Aggregate((d1, d2) => d1 * d2);

            // Transform from sparse to dense format, and flatten arrays.
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

            // one-hot encode targets for the classificaiton problem.
            var oneHotTargetsData = targetsData.EncodeOneHot();

            var targetsShape = new List<int>(outputShape);
            targetsShape.Add(observationCount);
            var targets = new Tensor(oneHotTargetsData, targetsShape.ToArray());

            return (observations, targets);
        }
    }
}
