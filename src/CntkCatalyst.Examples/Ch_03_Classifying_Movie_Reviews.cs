using System;
using System.IO;
using System.IO.Compression;
using System.Net;
using Accord.IO;
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

            var inputShape = new int[] { 1000 };
            var numberOfClasses = 1;
            var outputShape = new int[] { numberOfClasses };

            // download and load+vectorize imdb dataset.
            (var trainImages, var trainTargets) = LoadImdbData(inputShape, outputShape, DataSplit.Train);

            var network = new Sequential(Layers.Input(inputShape));

            network.Add(x => Layers.Dense(x, units: 16));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: 16));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: numberOfClasses));
            network.Add(x => Layers.Sigmoid(x));

            network.Compile(p => Learners.RMSProp(p, learningRate: 0.001),
               (t, p) => Losses.BinaryCrossEntropy(t, p),
               (t, p) => Metrics.Accuracy(t, p));

            // Split data and fit model with validation.
            
        }

        static (Tensor observations, Tensor targets) LoadImdbData(int[] inputShape, int[] outputShape, DataSplit dataSplit)
        {
            var xTrainFilePath = "x_train.npy";
            var yTrainFilePath = "y_train.npy";
            var xTestFilePath = "x_test.npy";
            var yTestFilePath = "y_test.npy";

            var imdbDataFilePath = "imdb.npz";
            
            if(!File.Exists(imdbDataFilePath))
            {
                using (var client = new WebClient())
                {
                    client.DownloadFile("https://s3.amazonaws.com/text-datasets/imdb.npz", "imdb.npz");
                }                
            }

            var observationsName = dataSplit == DataSplit.Train ? xTrainFilePath : xTestFilePath;
            var targetsName = dataSplit == DataSplit.Train ? yTrainFilePath : yTestFilePath;
            
            if(!File.Exists(observationsName))
            {
                ZipFile.ExtractToDirectory(imdbDataFilePath, ".");
            }

            var bytes = File.ReadAllBytes(observationsName);
            var observationsData = NpyFormat.LoadMatrix(bytes);
            //var targetsData = NpyFormat.LoadJagged(targetsName);

            throw new NotImplementedException();
        }
    }
}
