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

            // Define data type and device for the model.
            var d = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();

            // Create the architecture.
            var network = Layers.Input(inputShape, d)
                .Dense(16, d, device)
                .ReLU()
                .Dense(16, d, device)
                .ReLU()
                .Dense(numberOfClasses, d, device)
                .Softmax();

            // Create the network.
            var model = new Sequential(network, d, device);

            // Compile the network with the selected learner, loss and metric.
            model.Compile(p => Learners.Adam(p),
               (p, t) => Losses.BinaryCrossEntropy(p, t),
               (p, t) => Metrics.BinaryAccuracy(p, t));

            // Train the model using the training set.
            var history = model.Fit(partialTrainObservations, partialTrainTargets, epochs: 20, batchSize: 512, 
                xValidation: validationObservations, 
                yValidation: validationTargets);

            // Trace loss and validation history
            TraceLossValidationHistory(history);

            // Evaluate the model using the test set.
            (var loss, var metric) = model.Evaluate(testObservations, testTargets);

            // Write the test set loss and metric to debug output.
            Trace.WriteLine($"Test set - Loss: {loss}, Metric: {metric}");

            // Write first ten predictions
            var predictions = model.Predict(testObservations.GetSamples(Enumerable.Range(0, 10).ToArray()));

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
