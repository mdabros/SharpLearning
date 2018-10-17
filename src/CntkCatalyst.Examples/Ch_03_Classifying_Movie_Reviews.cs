using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CNTK;
using CntkCatalyst;
using CntkCatalyst.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Examples
{
    /// <summary>
    /// Example from Chapter 3.5: Classifying movie reviews:
    /// https://github.com/mdabros/deep-learning-with-python-notebooks/blob/master/3.5-classifying-movie-reviews.ipynb
    /// </summary>
    [TestClass]
    public class Ch_03_Classifying_Movie_Reviews
    {
        [TestMethod]
        public void Run()
        {
            // Define the input and output shape.
            var inputShape = new int[] { 10000 };
            var numberOfClasses = 1;
            var outputShape = new int[] { numberOfClasses };

            // Load train and test sets. 
            (var trainObservations, var trainTargets) = DataProvider
                .LoadImdbData(inputShape, outputShape, DataSplit.Train);
            (var testObservations, var testTargets) = DataProvider
                .LoadImdbData(inputShape, outputShape, DataSplit.Test);

            // Split validation set from train.
            var validationObservations = trainObservations.GetSamples(Enumerable.Range(0, 10000).ToArray());
            var validationTargets = trainTargets.GetSamples(Enumerable.Range(0, 10000).ToArray());
            var partialTrainObservations = trainObservations.GetSamples(Enumerable.Range(10000, 15000).ToArray());
            var partialTrainTargets = trainTargets.GetSamples(Enumerable.Range(10000, 15000).ToArray());

            // Create the network, and define the input shape.
            var d = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();
            var network = new Sequential(Layers.Input(inputShape), d, device);

            // Add layes to the network.
            network.Add(x => Layers.Dense(x, units: 16));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: 16));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: numberOfClasses));
            network.Add(x => Layers.Sigmoid(x));

            // Compile the network with the selected learner, loss and metric.
            network.Compile(p => Learners.Adam(p),
               (p, t) => Losses.BinaryCrossEntropy(p, t),
               (p, t) => Metrics.BinaryAccuracy(p, t));

            // Train the model using the training set.
            var history = network.Fit(partialTrainObservations, partialTrainTargets, epochs: 20, batchSize: 512, 
                xValidation: validationObservations, 
                yValidation: validationTargets);

            // Trace loss and validation history
            TraceLossValidationHistory(history);

            // Evaluate the model using the test set.
            (var loss, var metric) = network.Evaluate(testObservations, testTargets);

            // Write the test set loss and metric to debug output.
            Trace.WriteLine($"Test set - Loss: {loss}, Metric: {metric}");

            // Write first ten predictions
            var predictions = network.Predict(testObservations.GetSamples(Enumerable.Range(0, 10).ToArray()));

            // Use tensor data directly, since only 1 element pr. sample.
            Trace.WriteLine($"Predictions: [{string.Join(", ", predictions.Data)}]");          

            // TODO: Fix data download and parsing.
        }

        static void TraceLossValidationHistory(Dictionary<string, List<float>> history)
        {
            foreach (var item in history)
            {
                Trace.WriteLine($"{item.Key}: [{string.Join(", ", item.Value)}]");
            }
        }
    }
}
