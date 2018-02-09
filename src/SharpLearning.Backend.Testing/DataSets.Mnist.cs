namespace SharpLearning.Backend.Testing
{
    public static class DataSets
    {
        public static class Mnist
        {
            const string SourceUrl = "http://yann.lecun.com/exdb/mnist/";
            const string TrainImagesName = "train-images-idx3-ubyte.gz";
            const string TrainLabelsName = "train-labels-idx1-ubyte.gz";
            const string TestImagesName = "t10k-images-idx3-ubyte.gz";
            const string TestLabelsName = "t10k-labels-idx1-ubyte.gz";
            
            public static FlatTrainTestData<byte, byte> Load(string downloadDirectory)
            {
                var trainImages = DownloadAndLoad(TrainImagesName, downloadDirectory);
                var trainLabels = DownloadAndLoad(TrainLabelsName, downloadDirectory);
                var testImages = DownloadAndLoad(TestImagesName, downloadDirectory);
                var testLabels = DownloadAndLoad(TestLabelsName, downloadDirectory);
                return new FlatTrainTestData<byte, byte>(trainImages, trainLabels, testImages, testLabels);
            }

            private static FlatData<byte> DownloadAndLoad(string fileName, string downloadDirectory)
            {
                using (var s = Downloader.MaybeDownload(SourceUrl, fileName, downloadDirectory))
                {
                    var (shape, data) = IdxParser.ReadAll<byte>(s);
                    return new FlatData<byte>(shape, data);
                }
            }
        }
    }
}
