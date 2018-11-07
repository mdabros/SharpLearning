using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;

namespace CntkCatalyst.Examples
{
    public static class DataProvider
    {
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
