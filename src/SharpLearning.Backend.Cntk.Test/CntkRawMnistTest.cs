using System;
using System.IO;
using System.Linq;
using System.IO.Compression;
using System.Net;
using CNTK;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

// To avoid name-clash with SharpLearning.Backend.DataType.
using CntkDataType = CNTK.DataType;

namespace SharpLearning.Backend.Cntk.Test
{
    [TestClass]
    public class CntkRawMnistTest
    {
        const string DownloadPath = "MnistTest";

        [TestMethod]
        public void MnistSimpleExampleTest()
        {
            // Set global data and device type. 
            var dataType = CntkDataType.Float;
            DeviceDescriptor device = DeviceDescriptor.UseDefaultDevice();
            Trace.WriteLine("Using device: " + device.Type + " with data type: " + dataType);

            // Define data.
            var inputDimensions = 784;
            var numberOfClasses = 10;

            // Input variables denoting the features and label data.
            Variable x = Variable.InputVariable(new int[] { inputDimensions }, dataType); ;
            Variable y = Variable.InputVariable(new int[] { numberOfClasses }, dataType);

            // Model Parameters.
            Parameter w = new Parameter(new int[] { numberOfClasses, inputDimensions }, dataType, 0.0f, device, "W");
            Parameter b = new Parameter(new int[] { numberOfClasses }, dataType, 0.0f, device, "B");
            Function m = CNTKLib.Times(w, x); // variable and function are implicitly convertable
            
            // Linear Model.
            Function model = m + b;

            // Define loss and error metric.
            Function loss = CNTKLib.CrossEntropyWithSoftmax(model, y);
            Function errorMetric = CNTKLib.ClassificationError(model, y);

            // Training config.
            var minibatchSize = 64;
            var minibatchIterations = 200;

            // Instantiate progress writers.
            var trainingProgressOutputFreq = 100;

            // Instantiate the trainer object to drive the model training.
            var lr = new TrainingParameterScheduleDouble(0.01, 1);
            Trainer trainer = Trainer.CreateTrainer(model, loss, errorMetric, 
                new List<Learner>() { Learner.SGDLearner(model.Parameters(), lr) });

            // Load train data.
            var mnist = Mnist.Load(DownloadPath);
            var numberOftrainingSamples = mnist.TrainImages.Length;
            var readerTrain = mnist.GetTrainReader();

            // Train model.
            var inputMap = new Dictionary<Variable, Value>();
            for (int i = 0; i < minibatchIterations; i++)
            {
                // Using mnist.GetTrainReader().NextBatchArray(batchSize), which return the data as a single dim array.
                // the batch images and targets arrays should be reused to reduce allocations.
                (var trainingImages, var trainingTargets) = readerTrain.NextBatchArray(minibatchSize);

                // Note that it is possible to create a batch using a data buffer array, to reduce allocations. 
                // However, unsure how to handle random shuffling in this case.
                using (Value batchImages = Value.CreateBatch<float>(new int[] { inputDimensions }, trainingImages, device))
                using (Value batchTarget = Value.CreateBatch<float>(new int[] { numberOfClasses }, trainingTargets, device))
                {
                    inputMap.Add(x, batchImages);
                    inputMap.Add(y, batchTarget);

                    // There seems to be a memory-leak of some kind, like the batches are not properly released.
                    // This is most noticable with GPU training, since batches are being processed quicker.
                    trainer.TrainMinibatch(inputMap, false, device);
                    inputMap.Clear();

                    if ((i % trainingProgressOutputFreq) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
                    {
                        var batchLoss = (float)trainer.PreviousMinibatchLossAverage();
                        var batchError = (float)trainer.PreviousMinibatchEvaluationAverage();
                        Trace.WriteLine($"Minibatch: {i} CrossEntropyLoss = {batchLoss}, EvaluationCriterion = {batchError}");
                    }

                    var samplesSeenNextBatch = (i + 2) * minibatchSize;
                    if (samplesSeenNextBatch >= numberOftrainingSamples)
                    {
                        readerTrain.Reset();
                    }
                }
            }

            // Load test data.
            var readerTest = mnist.GetTrainReader();

            // Test data for trained model.
            var testMinibatchSize = 1024;
            var numberOfTestSamples = mnist.TestImages.Length;
            var numMinibatchesToTest = numberOfTestSamples / testMinibatchSize;
            var testResult = 0.0;

            var testInputMap = new UnorderedMapVariableMinibatchData();
            for (int i = 0; i < numMinibatchesToTest; i++)
            {
                // Using mnist.GetTrainReader().NextBatchArray(batchSize), which return the data as a single dim array.
                // the batch images and targets arrays should be reused to reduce allocations.
                (var testImages, var testTargets) = readerTest.NextBatchArray(minibatchSize);

                // Note that it is possible to create a batch using a data buffer array, to reduce allocations. 
                // However, unsure how to handle random shuffling in this case.
                using (Value batchImages = Value.CreateBatch<float>(new int[] { inputDimensions }, testImages, device))
                using (Value batchTarget = Value.CreateBatch<float>(new int[] { numberOfClasses }, testTargets, device))
                {
                    testInputMap.Add(x, new MinibatchData(batchImages));
                    testInputMap.Add(y, new MinibatchData(batchTarget));

                    // There seems to be a memory-leak of some kind, like the batches are not properly released.
                    // This is most noticable with GPU training, since batches are being processed quicker.
                    var evalError = trainer.TestMinibatch(testInputMap);
                    testInputMap.Clear();

                    testResult += evalError;
                }
            }

            var csharpError = testResult / numMinibatchesToTest;
            Trace.WriteLine($"Test Error: {csharpError}");

            var pythonError = 0.124; // Most likely from diffent observations, both in training and in test.
            Assert.AreEqual(pythonError, csharpError, 0.00001);
        }

        [TestMethod]
        public void MnistCnnExampleTest()
        {
            // Define data.
            var imageHeight = 28;
            var imageWidth = 28;
            var numChannels = 1;
            var input_dim = imageHeight * imageWidth * numChannels;
            var numOutputClasses = 10;
            var dataType = CntkDataType.Float;
            var inputDimensions = new int[] { numChannels, imageHeight, imageWidth };

            // Set device.
            var device = DeviceDescriptor.UseDefaultDevice();

            // Input variables denoting the features and label data.
            Variable inputVar = Variable.InputVariable(inputDimensions, dataType); ;
            Variable labelVar = Variable.InputVariable(new int[] { numOutputClasses }, dataType);

            // Instantiate the feedforward classification model
            var scaledInput = CNTKLib.ElementTimes(Constant.Scalar(dataType, 0.00390625, device), inputVar);

            var layers = new CntkLayers(device, dataType);
            Function z = layers.Dense(scaledInput, numOutputClasses);

            // Define loss and error metric.
            Function loss = CNTKLib.CrossEntropyWithSoftmax(z, labelVar);
            Function errorMetric = CNTKLib.ClassificationError(z, labelVar);

            // Set learning parameters
            var lrSchedule = new TrainingParameterScheduleDouble(0.01, 1);
            var mmSchedule = new TrainingParameterScheduleDouble(0.9990239141819757, 1);

            // Instantiate the trainer object to drive the model training
            var learner = new List<Learner>() { Learner.MomentumSGDLearner(z.Parameters(), lrSchedule, mmSchedule, false) };
            //var learner = new List<Learner>() { Learner.SGDLearner(z.Parameters(), lrSchedule) };
            var trainingProgressOutputFreq = 100;
            var trainer = Trainer.CreateTrainer(z, loss, errorMetric, learner);

            // Training config.
            var mnist = Mnist.Load(DownloadPath);
            var readerTrain = mnist.GetTrainReader();

            var minibatchSize = 64;
            var epochSize = mnist.TrainImages.Count();
            var maxEpochs = 40;
            var minibatchIterations = 2000;

            // Train model.
            var inputMap = new Dictionary<Variable, Value>();
            for (int i = 0; i < minibatchIterations; i++)
            {
                // Using mnist.GetTrainReader().NextBatchArray(batchSize), which return the data as a single dim array.
                // the batch images and targets arrays should be reused to reduce allocations.
                (var trainingImages, var trainingTargets) = readerTrain.NextBatchArray(minibatchSize);

                // Note that it is possible to create a batch using a data buffer array, to reduce allocations. 
                // However, unsure how to handle random shuffling in this case.
                using (Value batchImages = Value.CreateBatch<float>(inputDimensions, trainingImages, device))
                using (Value batchTarget = Value.CreateBatch<float>(new int[] { numOutputClasses }, trainingTargets, device))
                {
                    inputMap.Add(inputVar, batchImages);
                    inputMap.Add(labelVar, batchTarget);

                    // There seems to be a memory-leak of some kind, like the batches are not properly released.
                    // This is most noticable with GPU training, since batches are being processed quicker.
                    trainer.TrainMinibatch(inputMap, false, device);
                    inputMap.Clear();

                    if ((i % trainingProgressOutputFreq) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
                    {
                        var batchLoss = (float)trainer.PreviousMinibatchLossAverage();
                        var batchError = (float)trainer.PreviousMinibatchEvaluationAverage();
                        Trace.WriteLine($"Minibatch: {i} CrossEntropyLoss = {batchLoss}, EvaluationCriterion = {batchError}");
                    }

                    var samplesSeenNextBatch = (i + 2) * minibatchSize;
                    if (samplesSeenNextBatch >= epochSize)
                    {
                        readerTrain.Reset();
                    }
                }
            }
        }
    }

    public enum Activation
    {
        None,
        ReLU,
        LeakyReLU,
        Sigmoid,
        Tanh
    }

    // Below copied from: https://github.com/migueldeicaza/TensorFlowSharp/blob/master/Learn/Datasets/MNIST.cs
    // Do NOT use this code as foundation for other things, quality is not great!

    // Stores the per-image MNIST information we loaded from disk 
    //
    // We store the data in two formats, byte array (as it came in from disk), and float array
    // where each 0..255 value has been mapped to 0.0f..1.0f
    public struct MnistImage
    {
        public int Cols, Rows;
        public byte[] Data;
        public float[] DataFloat;

        public MnistImage(int cols, int rows, byte[] data)
        {
            Cols = cols;
            Rows = rows;
            Data = data;
            DataFloat = new float[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                DataFloat[i] = Data[i] / 255f;
            }
        }
    }

    // Helper class used to load and work with the Mnist data set
    public class Mnist
    {
        // 
        // The loaded results
        //
        public MnistImage[] TrainImages, TestImages, ValidationImages;
        public byte[] TrainLabels, TestLabels, ValidationLabels;
        public byte[,] OneHotTrainLabels, OneHotTestLabels, OneHotValidationLabels;

        public BatchReader GetTrainReader(int readFromItem) => new BatchReader(TrainImages, TrainLabels, OneHotTrainLabels, readFromItem);
        public BatchReader GetTrainReader() => new BatchReader(TrainImages, TrainLabels, OneHotTrainLabels);
        public BatchReader GetTestReader() => new BatchReader(TestImages, TestLabels, OneHotTestLabels);
        public BatchReader GetValidationReader() => new BatchReader(ValidationImages, ValidationLabels, OneHotValidationLabels);

        public class BatchReader
        {
            int start = 0;
            MnistImage[] source;
            byte[] labels;
            byte[,] oneHotLabels;

            internal BatchReader(MnistImage[] source, byte[] labels, byte[,] oneHotLabels, int readFromItem = 0)
            {
                this.source = source;
                this.labels = labels;
                this.oneHotLabels = oneHotLabels;
                start = readFromItem;
            }

            public (float[,], float[,]) NextBatch(int batchSize)
            {
                // TODO: Remove consts and allocs...
                var imageData = new float[batchSize, 784];
                var labelData = new float[batchSize, 10];

                int p = 0;
                for (int item = 0; item < batchSize; item++)
                {
                    Buffer.BlockCopy(source[start + item].DataFloat, 0, imageData, p, 784 * sizeof(float));
                    p += 784 * sizeof(float);
                    for (var j = 0; j < 10; j++)
                        labelData[item, j] = oneHotLabels[item + start, j];
                }

                start += batchSize;
                return (imageData, labelData);
            }

            public (float[], float[]) NextBatchArray(int batchSize)
            {
                // TODO: Remove consts and allocs...
                var imageData = new float[batchSize * 784];
                var labelData = new float[batchSize * 10];

                int p = 0;
                for (int item = 0; item < batchSize; item++)
                {
                    Buffer.BlockCopy(source[start + item].DataFloat, 0, imageData, p, 784 * sizeof(float));
                    p += 784 * sizeof(float);
                    for (var j = 0; j < 10; j++)
                    {
                        var rowOffSet = item * 10;
                        labelData[rowOffSet + j] = oneHotLabels[item + start, j];
                    }
                }

                start += batchSize;
                return (imageData, labelData);
            }

            public void Reset()
            {
                start = 0;
            }
        }

        int Read32(Stream s)
        {
            var x = new byte[4];
            s.Read(x, 0, 4);
            var bigEndian = BitConverter.ToInt32(x, 0);
            return BigEndianToInt32(bigEndian);// DataConverter.BigEndian.GetInt32(x, 0);
        }

        int BigEndianToInt32(int bigEndian)
        {
            if (BitConverter.IsLittleEndian)
            {
                return (int)SwapBytes((uint)bigEndian);
            }
            return bigEndian;
        }

        public ushort SwapBytes(ushort x)
        {
            return (ushort)((ushort)((x & 0xff) << 8) | ((x >> 8) & 0xff));
        }

        public uint SwapBytes(uint x)
        {
            return ((x & 0x000000ff) << 24) +
                   ((x & 0x0000ff00) << 8) +
                   ((x & 0x00ff0000) >> 8) +
                   ((x & 0xff000000) >> 24);
        }

        public ulong SwapBytes(ulong x)
        {
            // swap adjacent 32-bit blocks
            x = (x >> 32) | (x << 32);
            // swap adjacent 16-bit blocks
            x = ((x & 0xFFFF0000FFFF0000) >> 16) | ((x & 0x0000FFFF0000FFFF) << 16);
            // swap adjacent 8-bit blocks
            return ((x & 0xFF00FF00FF00FF00) >> 8) | ((x & 0x00FF00FF00FF00FF) << 8);
        }

        MnistImage[] ExtractImages(Stream input, string file)
        {
            using (var gz = new GZipStream(input, CompressionMode.Decompress))
            {
                if (Read32(gz) != 2051)
                    throw new Exception("Invalid magic number found on the MNIST " + file);
                var count = Read32(gz);
                var rows = Read32(gz);
                var cols = Read32(gz);

                var result = new MnistImage[count];
                for (int i = 0; i < count; i++)
                {
                    var size = rows * cols;
                    var data = new byte[size];
                    gz.Read(data, 0, size);

                    result[i] = new MnistImage(cols, rows, data);
                }
                return result;
            }
        }


        byte[] ExtractLabels(Stream input, string file)
        {
            using (var gz = new GZipStream(input, CompressionMode.Decompress))
            {
                if (Read32(gz) != 2049)
                    throw new Exception("Invalid magic number found on the MNIST " + file);
                var count = Read32(gz);
                var labels = new byte[count];
                gz.Read(labels, 0, count);

                return labels;
            }
        }

        T[] Pick<T>(T[] source, int first, int last)
        {
            if (last == 0)
                last = source.Length;
            var count = last - first;
            var result = new T[count];
            Array.Copy(source, first, result, 0, count);
            return result;
        }

        // Turn the labels array that contains values 0..numClasses-1 into
        // a One-hot encoded array
        byte[,] OneHot(byte[] labels, int numClasses)
        {
            var oneHot = new byte[labels.Length, numClasses];
            for (int i = 0; i < labels.Length; i++)
            {
                oneHot[i, labels[i]] = 1;
            }
            return oneHot;
        }

        /// <summary>
        /// Reads the data sets.
        /// </summary>
        /// <param name="trainDir">Directory where the training data is downlaoded to.</param>
        /// <param name="numClasses">Number classes to use for one-hot encoding, or zero if this is not desired</param>
        /// <param name="validationSize">Validation size.</param>
        public void ReadDataSets(string trainDir, int numClasses = 10, int validationSize = 5000)
        {
            const string SourceUrl = "http://yann.lecun.com/exdb/mnist/";
            const string TrainImagesName = "train-images-idx3-ubyte.gz";
            const string TrainLabelsName = "train-labels-idx1-ubyte.gz";
            const string TestImagesName = "t10k-images-idx3-ubyte.gz";
            const string TestLabelsName = "t10k-labels-idx1-ubyte.gz";

            TrainImages = ExtractImages(Helper.MaybeDownload(SourceUrl, trainDir, TrainImagesName), TrainImagesName);
            TestImages = ExtractImages(Helper.MaybeDownload(SourceUrl, trainDir, TestImagesName), TestImagesName);
            TrainLabels = ExtractLabels(Helper.MaybeDownload(SourceUrl, trainDir, TrainLabelsName), TrainLabelsName);
            TestLabels = ExtractLabels(Helper.MaybeDownload(SourceUrl, trainDir, TestLabelsName), TestLabelsName);

            ValidationImages = Pick(TrainImages, 0, validationSize);
            ValidationLabels = Pick(TrainLabels, 0, validationSize);
            TrainImages = Pick(TrainImages, validationSize, 0);
            TrainLabels = Pick(TrainLabels, validationSize, 0);

            if (numClasses != -1)
            {
                OneHotTrainLabels = OneHot(TrainLabels, numClasses);
                OneHotValidationLabels = OneHot(ValidationLabels, numClasses);
                OneHotTestLabels = OneHot(TestLabels, numClasses);
            }
        }

        public static Mnist Load(string downloadPath)
        {
            var x = new Mnist();
            x.ReadDataSets(downloadPath);
            return x;
        }
    }

    public class Helper
    {
        public static Stream MaybeDownload(string urlBase, string trainDir, string file)
        {
            if (!Directory.Exists(trainDir))
                Directory.CreateDirectory(trainDir);
            var target = Path.Combine(trainDir, file);
            if (!File.Exists(target))
            {
                var wc = new WebClient();
                wc.DownloadFile(urlBase + file, target);
            }
            return File.OpenRead(target);
        }
    }
}
