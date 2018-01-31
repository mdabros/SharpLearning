using System;
using System.IO;
using System.Linq;
using System.IO.Compression;
using System.Net;
using CNTK;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.Diagnostics;

// To avoid name-clash with SharpLearning.Backend.DataType.
using CntkDataType = CNTK.DataType;
using System.Text;

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
            var minibatchIterations = 2000;

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

            Trace.WriteLine($"Test Error: {testResult / numMinibatchesToTest}");
        }

        [TestMethod]
        public void MnistCnnExampleTest()
        {
            // Data set
            var mnist = Mnist.Load(DownloadPath);
            var trainingSampleCount = mnist.TrainImages.Length;
            var dataShape = new int[] { 28, 28, 1 };
            var numberOfClasses = 10;
            var outputValidationError = false;

            // Set global data and device types. 
            var dataType = CntkDataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();
            Trace.WriteLine("Using device: " + device.Type);
            // These are just hardcoded in the helper class.
            CntkLayers.DataType = dataType;
            CntkLayers.Device = device;
           
            // Variables for setting features and targets while training the network.
            var featureVariable = Variable.InputVariable(dataShape, dataType);
            var targetVariable = Variable.InputVariable(new int[] { numberOfClasses }, dataType);

            // Define small convolutional neural network.
            Function conv2d = CntkLayers.Conv2D(featureVariable, 3, 3, 32);
            Function conv2dActivation = CntkLayers.ActivationFunction(conv2d, Activation.ReLU);

            Function pool2d = CntkLayers.Pool2D(conv2dActivation, 2, 2);

            Function dense = CntkLayers.Dense(pool2d, 256);
            Function denseActivation = CntkLayers.ActivationFunction(dense, Activation.ReLU);

            Function dropout = CntkLayers.Dropout(denseActivation, 0.5);

            Function cnn = CntkLayers.Dense(dropout, numberOfClasses);

            // Training loss and eval error. Eval error is only for reporting and not used for training.
            var loss = CNTKLib.CrossEntropyWithSoftmax(cnn, targetVariable);
            var evalError = CNTKLib.ClassificationError(cnn, targetVariable);

            // Setup stochastic gradient descent learner
            var learningRatePerSample = new TrainingParameterScheduleDouble(0.01, 1);
            var parameterLearners = new List<Learner>() { Learner.SGDLearner(cnn.Parameters(), learningRatePerSample) };

            // Setup trainer with the defined network, loss and optimizer.
            Trainer trainer = Trainer.CreateTrainer(cnn, loss, evalError, parameterLearners);

            var epochs = 100; // how many epochs to train.
            var batchSize = 64; // size of each training batch
            var numberOfBatchesPrEpoch = trainingSampleCount / batchSize; // rough calculation of how many batches pr. epoch
            var random = new Random(232);
            var timer = new Stopwatch();

            // Train the model
            var batchContainer = new Dictionary<Variable, Value>();
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Reset accumulations for loss and error metric for each epoch
                var accumulatedLoss = 0.0;
                var accumulatedClassificationError = 0.0;

                timer.Restart();

                for (int i = 0; i < numberOfBatchesPrEpoch; i++)
                {
                    // Using mnist.GetTrainReader().NextBatchArray(batchSize), which return the data as a single dim array.
                    // After each epoch, the training data must be shuffled.
                    // Emulate this by starting the batch at a random index. This is not optimal.
                    var randomBatchStart = random.Next(0, trainingSampleCount - batchSize);
                    // the batch images and targets arrays should be reused to reduce allocations.
                    (var trainingImages, var trainingTargets) = mnist.GetTrainReader(readFromItem: randomBatchStart)
                        .NextBatchArray(batchSize);

                    // Note that it is possible to create a batch using a data buffer array, to reduce allocations. 
                    // However, unsure how to handle random shuffling in this case.
                    using (Value batchImages = Value.CreateBatch<float>(dataShape, trainingImages, device))
                    using (Value batchTarget = Value.CreateBatch<float>(new int[] { numberOfClasses }, trainingTargets, device))
                    {
                        batchContainer.Add(featureVariable, batchImages);
                        batchContainer.Add(targetVariable, batchTarget);

                        // There seems to be a memory-leak of some kind, like the batches are not properly released.
                        // This is most noticable with GPU training, since batches are being processed quicker.
                        trainer.TrainMinibatch(batchContainer, false, device);
                        batchContainer.Clear();

                        accumulatedLoss += trainer.PreviousMinibatchSampleCount() * trainer.PreviousMinibatchLossAverage();
                        accumulatedClassificationError += trainer.PreviousMinibatchSampleCount() * trainer.PreviousMinibatchEvaluationAverage();
                    }
                }
                timer.Stop();

                var currentLoss = accumulatedLoss / (double)trainingSampleCount;
                var currentError = accumulatedClassificationError / (double)trainingSampleCount;
                var epochTime_ms = timer.ElapsedMilliseconds;

                // report results.
                var output = new StringBuilder();
                output.AppendLine($"Epoch { epoch + 1:000}:");
                output.AppendLine($"   Train: Error: {currentError:F12}, Loss: {currentLoss:F12}, Time (ms): {epochTime_ms}");

                if (outputValidationError)
                {
                    timer.Restart();
                    var validationError = Validate(cnn, mnist, targetVariable, featureVariable);
                    timer.Stop();
                    var totalValidationTime_ms = timer.ElapsedMilliseconds;
                    output.AppendLine($"   Valid: Error: {validationError:F12}, Samples: {mnist.ValidationImages.Length}, Time (ms): {totalValidationTime_ms}");
                }

                Trace.Write(output);

                // Sometimes it seems the GC is too busy to collect.
                // Might be caused by native memory not visible for the managed gc
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        public double Validate(Function model, Mnist mnist,
            Variable labelVariable, Variable featureVariable)
        {
            var samples = mnist.ValidationImages.Length;
            var reader = mnist.GetValidationReader();

            var validationError = 0.0;
            // More efficient to batch samples, however, run separate to get worst case.
            for (int i = 0; i < samples; i++)
            {
                (var image, var target) = reader.NextBatchArray(1);
                var inputDimensions = model.Arguments[0].Shape;

                using (Value batch = Value.CreateBatch<float>(inputDimensions, image, CntkLayers.Device))
                {
                    Variable actualVariable = CNTKLib.InputVariable(labelVariable.Shape, CntkLayers.DataType);
                    Function evalMetricFunc = CNTKLib.ClassificationError(labelVariable, actualVariable);
                    Value actual = Evaluate(model, featureVariable, image, inputDimensions);
                    Value expected = Value.CreateBatch(new int[] { target.Length }, target, CntkLayers.Device);

                    var inputDataMap = new Dictionary<Variable, Value>() { { labelVariable, expected }, { actualVariable, actual } };
                    var outputDataMap = new Dictionary<Variable, Value>() { { evalMetricFunc.Output, null } };
                    evalMetricFunc.Evaluate(inputDataMap, outputDataMap, CntkLayers.Device);

                    List<float> evalMetric = outputDataMap[evalMetricFunc.Output].GetDenseData<float>(evalMetricFunc.Output).Select(x => x.First()).ToList();
                    validationError += evalMetric.Average();
                }
            }

            validationError = validationError / (double)samples;
            return validationError;
        }

        Value Evaluate(Function model, Variable featureVariable,
            float[] image, NDShape imageDimensions)
        {
            Value features = Value.CreateBatch(imageDimensions, image, CntkLayers.Device);

            var inputDataMap = new Dictionary<Variable, Value>() { { featureVariable, features } };
            var outputDataMap = new Dictionary<Variable, Value>() { { model.Output, null } };

            model.Evaluate(inputDataMap, outputDataMap, CntkLayers.Device);
            return outputDataMap[model.Output];
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

    /// <summary>
    /// Helper class to make CNTK operator creation more simple.
    /// </summary>
    public static class CntkLayers
    {
        public static DeviceDescriptor Device = DeviceDescriptor.UseDefaultDevice();
        public static CntkDataType DataType = CntkDataType.Float;

        public static Function Input(params int[] inputDim)
        {
            return Variable.InputVariable(inputDim, DataType);
        }

        public static Function SoftMax(Variable input)
        {
            return CNTKLib.Softmax(input);
        }

        public static Function Dense(Variable input, int units, uint seed = 32, string outputName = "")
        {
            if (input.Shape.Rank != 1)
            {
                // Flatten dimensions.
                var newDim = input.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                input = CNTKLib.Reshape(input, new int[] { newDim });
            }

            // Use GlorotUniform for weight initialization.
            var initializer = CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, seed);

            var inputDim = input.Shape[0];

            var weights = new Parameter(new int[] { units, inputDim },
                DataType, initializer, Device, "timesParam");

            // Bias is initialized to 0.0.
            var bias = new Parameter(new int[] { units }, 0.0f, Device, "plusParam");

            return CNTKLib.Times(weights, input) + bias;
        }

        public static Function ActivationFunction(Variable input, Activation activation)
        {
            switch (activation)
            {
                default:
                case Activation.None:
                    return input;
                case Activation.ReLU:
                    return CNTKLib.ReLU(input);
                case Activation.LeakyReLU:
                    return CNTKLib.LeakyReLU(input);
                case Activation.Sigmoid:
                    return CNTKLib.Sigmoid(input);
                case Activation.Tanh:
                    return CNTKLib.Tanh(input);
            }
        }

        public static Function Conv2D(Variable input, int filterW, int filterH, int filterCount,
             int strideW = 1, int strideH = 1, uint seed = 34, string outputName = "")
        {
            if (input.Shape.Rank != 3)
            {
                throw new ArgumentException("Conv2D layer requires shape rank 3, got rank " + input.Shape.Rank);
            }

            // Assumes specific layout.
            var inputChannels = input.Shape[2];

            // Use GlorotUniform for weight initialization.
            var intializer = CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, seed);

            var convParams = new Parameter(new int[] { filterW, filterH, inputChannels, filterCount },
                    DataType, intializer, Device);
            var conv = CNTKLib.Convolution(convParams, input, new int[] { strideW, strideH, inputChannels });

            // Bias is initialized to 0.0.
            var bias = new Parameter(conv.Output.Shape, DataType, 0.0, Device);
            return CNTKLib.Plus(bias, conv);
        }

        public static Function Pool2D(Variable input, int poolW, int poolH,
            PoolingType poolingType = PoolingType.Max,
            int strideW = 2, int strideH = 2, string outputName = "")
        {
            if (input.Shape.Rank != 3)
            {
                throw new ArgumentException("Pool2D layer requires shape rank 3, got rank " + input.Shape.Rank);
            }

            return CNTKLib.Pooling(input, PoolingType.Max,
                new int[] { poolW, poolH }, new int[] { strideW, strideH });
        }

        public static Function Reshape(Variable layer, int[] targetShape)
        {
            return CNTKLib.Reshape(layer, targetShape);
        }

        public static Function GlobalAveragePool2D(Variable layer)
        {
            return CNTKLib.Pooling(layer, PoolingType.Average, new int[] { layer.Shape[0], layer.Shape[1] });
        }

        public static Function Dropout(Variable input, double dropoutRate, uint seed = 465)
        {
            return CNTKLib.Dropout(input, dropoutRate, seed);
        }

        public static Function BatchNormalizationLayer(Variable input, bool spatial,
            double initialScaleValue = 1, double initialBiasValue = 0, int bnTimeConst = 5000)
        {
            var biasParams = new Parameter(new int[] { NDShape.InferredDimension }, (float)initialBiasValue, Device, "");
            var scaleParams = new Parameter(new int[] { NDShape.InferredDimension }, (float)initialScaleValue, Device, "");
            var runningMean = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, Device);
            var runningInvStd = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, Device);
            var runningCount = Constant.Scalar(0.0f, Device);

            return CNTKLib.BatchNormalization(input, scaleParams, biasParams, runningMean, runningInvStd, runningCount,
                spatial, (double)bnTimeConst, 0.0, 1e-5 /* epsilon */);
        }
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
