using System.Diagnostics;
using CNTK;
using CntkCatalyst;
using CntkCatalyst.Models;
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
            (var trainImages, var trainTargets) = DataProvider
                .LoadMnistData(inputShape, outputShape, DataSplit.Train);
            (var testImages, var testTargets) = DataProvider
                .LoadMnistData(inputShape, outputShape, DataSplit.Test);

            // Define data type and device for the model.
            var d = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();

            // Create the architecture.
            var network = Layers.Input(inputShape, d)
                .Dense(512, d, device)
                .ReLU()
                .Dense(numberOfClasses, d, device)
                .Softmax();

            // Create the network.
            var model = new Sequential(network, d, device);

            // Compile the network with the selected learner, loss and metric.
            model.Compile(p => Learners.RMSProp(p),
               (p, t) => Losses.CategoricalCrossEntropy(p, t),
               (p, t) => Metrics.Accuracy(p, t));

            // Train the model using the training set.
            model.Fit(trainImages, trainTargets, epochs: 5, batchSize: 128);

            // Evaluate the model using the test set.
            (var loss, var metric) = model.Evaluate(testImages, testTargets);

            // Write the test set loss and metric to debug output.
            Trace.WriteLine($"Test set - Loss: {loss}, Metric: {metric}");
        }
    }
}
