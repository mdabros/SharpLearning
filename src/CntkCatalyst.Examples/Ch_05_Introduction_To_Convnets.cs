using System.Diagnostics;
using CNTK;
using CntkCatalyst;
using CntkCatalyst.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Examples
{
    /// <summary>
    /// Example from Chapter 5.1: Introduction to convnets:
    /// https://github.com/mdabros/deep-learning-with-python-notebooks/blob/master/5.1-introduction-to-convnets.ipynb
    /// </summary>
    [TestClass]
    public class Ch_05_Introduction_To_Convnets
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
            (var trainImages, var trainTargets) = DataProvider
                .LoadMnistData(inputShape, outputShape, DataSplit.Train);
            (var testImages, var testTargets) = DataProvider
                .LoadMnistData(inputShape, outputShape, DataSplit.Test);

            // Create the network, and define the input shape.
            var network = new Sequential(Layers.Input(inputShape));

            // Add layes to the network.
            network.Add(x => Layers.Conv2D(x, 3, 3, 32));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Pool2D(x, 2, 2, PoolingType.Max));

            network.Add(x => Layers.Conv2D(x, 3, 3, 64));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Pool2D(x, 2, 2, PoolingType.Max));

            network.Add(x => Layers.Conv2D(x, 3, 3, 64));
            network.Add(x => Layers.ReLU(x));

            network.Add(x => Layers.Dense(x, units: 64));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: numberOfClasses));
            network.Add(x => Layers.Softmax(x));

            // Compile the network with the selected learner, loss and metric.
            network.Compile(p => Learners.Adam(p),
               (p, t) => Losses.CategoricalCrossEntropy(p, t),
               (p, t) => Metrics.Accuracy(p, t));

            // Train the model using the training set.
            network.Fit(trainImages, trainTargets, epochs: 5, batchSize: 64);

            // Evaluate the model using the test set.
            (var loss, var metric) = network.Evaluate(testImages, testTargets);

            // Write the test set loss and metric to debug output.
            Trace.WriteLine($"Test set - Loss: {loss}, Metric: {metric}");

            // Write model summary.
            Trace.WriteLine(network.Summary());
        }
    }
}
