using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using CntkExtensions;
using CntkExtensions.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Examples
{
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
            (var trainObservations, var trainTargets) = LoadImdbData(inputShape, outputShape, DataSplit.Train);
            (var testObservations, var testTargets) = LoadImdbData(inputShape, outputShape, DataSplit.Test);

            // Split validation set from train.
            var validationObservations = trainObservations.GetSamples(Enumerable.Range(0, 10000).ToArray());
            var validationTargets = trainTargets.GetSamples(Enumerable.Range(0, 10000).ToArray());
            var partialTrainObservations = trainObservations.GetSamples(Enumerable.Range(10000, 15000).ToArray());
            var partialTrainTargets = trainTargets.GetSamples(Enumerable.Range(10000, 15000).ToArray());

            // Create the network, and define the input shape.
            var network = new Sequential(Layers.Input(inputShape));

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

            // TODO: Plot history?.
            // TODO: Fix data download and parsing.
        }

        static (Tensor observations, Tensor targets) LoadImdbData(int[] inputShape, int[] outputShape, DataSplit dataSplit)
        {
            var xTrainFilePath = "x_train.bin";
            var yTrainFilePath = "y_train.bin";
            var xTestFilePath = "x_test.bin";
            var yTestFilePath = "y_test.bin";

            var imdbDataFilePath = "imdb_data.zip";
            
            if(!File.Exists(imdbDataFilePath))
            {
                // TODO: Fix download of correct file.
                using (var client = new WebClient())
                {
                    client.DownloadFile("https://s3.amazonaws.com/text-datasets/imdb.npz", "imdb_data.zip");
                }                
            }

            var observationsName = dataSplit == DataSplit.Train ? xTrainFilePath : xTestFilePath;
            var targetsName = dataSplit == DataSplit.Train ? yTrainFilePath : yTestFilePath;
            
            if(!File.Exists(observationsName))
            {
                ZipFile.ExtractToDirectory(imdbDataFilePath, ".");
            }

            var observationCount = 25000;
            var featureCount = inputShape.Single();

            var observationsData = LoadBinaryFile(observationsName, observationCount * featureCount);          
            var observations = new Tensor(observationsData, inputShape.ToArray(), observationCount);

            var targetsData = LoadBinaryFile(targetsName, observationCount);         
            var targets = new Tensor(targetsData, outputShape.ToArray(), observationCount);

            return (observations, targets);
        }

        static float[] LoadBinaryFile(string filepath, int N)
        {
            var buffer = new byte[sizeof(float) * N];
            using (var reader = new System.IO.BinaryReader(System.IO.File.OpenRead(filepath)))
            {
                reader.Read(buffer, 0, buffer.Length);
            }
            var dst = new float[N];
            System.Buffer.BlockCopy(buffer, 0, dst, 0, buffer.Length);
            return dst;
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
