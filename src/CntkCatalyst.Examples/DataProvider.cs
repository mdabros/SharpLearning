using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using Accord.DataSets;
using CntkCatalyst;

namespace CntkCatalyst.Examples
{
    public static class DataProvider
    {
        public static (MemoryMinibatchData observations, MemoryMinibatchData targets) LoadMnistData(int[] inputShape, int[] outputShape, DataSplit dataSplit)
        {
            // Load mnist data set using Accord.DataSets.
            var mnist = new MNIST(Directory.GetCurrentDirectory());
            var dataSet = dataSplit == DataSplit.Train ? mnist.Training : mnist.Testing;

            var sampleCount = dataSet.Item2.Length;
            var dataSize = inputShape.Aggregate((d1, d2) => d1 * d2);

            // Transform from sparse to dense format, and flatten arrays.
            var observationsData = dataSet.Item1
                .Select(s => s.ToDense(dataSize)) // set fixed dataSize.
                .SelectMany(d => d)
                .Select(d => (float)d / 255) // transform pixel values to be between 0 and 1.
                .ToArray();

            var targetsData = dataSet.Item2
                .Select(d => (float)d)
                .ToArray();

            var observations = new MemoryMinibatchData(observationsData, inputShape.ToArray(), sampleCount);

            // one-hot encode targets for the classificaiton problem.
            var oneHotTargetsData = targetsData.EncodeOneHot();
            var targets = new MemoryMinibatchData(oneHotTargetsData, outputShape.ToArray(), sampleCount);

            return (observations, targets);
        }

        public static (MemoryMinibatchData observations, MemoryMinibatchData targets) LoadImdbData(int[] inputShape, int[] outputShape, DataSplit dataSplit)
        {
            var xTrainFilePath = "x_train.bin";
            var yTrainFilePath = "y_train.bin";
            var xTestFilePath = "x_test.bin";
            var yTestFilePath = "y_test.bin";

            var imdbDataFilePath = "imdb_data.zip";

            if (!File.Exists(imdbDataFilePath))
            {
                // TODO: Fix download of correct file.
                using (var client = new WebClient())
                {
                    client.DownloadFile("https://s3.amazonaws.com/text-datasets/imdb.npz", "imdb_data.zip");
                }
            }

            var observationsName = dataSplit == DataSplit.Train ? xTrainFilePath : xTestFilePath;
            var targetsName = dataSplit == DataSplit.Train ? yTrainFilePath : yTestFilePath;

            if (!File.Exists(observationsName))
            {
                ZipFile.ExtractToDirectory(imdbDataFilePath, ".");
            }

            var observationCount = 25000;
            var featureCount = inputShape.Single();

            var observationsData = LoadBinaryFile(observationsName, observationCount * featureCount);
            var observations = new MemoryMinibatchData(observationsData, inputShape.ToArray(), observationCount);

            var targetsData = LoadBinaryFile(targetsName, observationCount);
            var targets = new MemoryMinibatchData(targetsData, outputShape.ToArray(), observationCount);

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
    }
}
