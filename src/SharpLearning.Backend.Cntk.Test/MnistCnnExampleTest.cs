using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CNTK;
using Microsoft.VisualStudio.TestTools.UnitTesting;
// To avoid name-clash with SharpLearning.Backend.DataType.
using CntkDataType = CNTK.DataType;

namespace SharpLearning.Backend.Cntk.Test
{
    [TestClass]
    public class MnistCnnExampleTest
    {
        const string DownloadPath = "MnistTest";
             
        [TestMethod]
        public void MnistCnnTest()
        {
            // Set global data and device type. 
            var dataType = CntkDataType.Float;
            DeviceDescriptor device = DeviceDescriptor.UseDefaultDevice();
            Trace.WriteLine("Using device: " + device.Type + " with data type: " + dataType);

            // Define data.
            var imageHeight = 28;
            var imageWidth = 28;
            var numChannels = 1;
            var numOutputClasses = 10;
            var inputDimensions = new int[] { imageWidth, imageHeight, numChannels };

            // Input variables denoting the features and label data.
            Variable inputVar = Variable.InputVariable(inputDimensions, dataType); ;
            Variable labelVar = Variable.InputVariable(new int[] { numOutputClasses }, dataType);

            // Instantiate the feedforward classification model
            var scaledInput = CNTKLib.ElementTimes(Constant.Scalar(dataType, 0.00390625, device), inputVar);

            var layers = new CntkLayers(device, dataType);
            Function conv1 = layers.Convolution2D(scaledInput, new int[] { 5, 5 }, 32, Activation.Relu, pad: true);
            Function pool1 = layers.MaxPooling(conv1, new int[] { 3, 3 }, new int[] { 2, 2 });
            Function conv2 = layers.Convolution2D(pool1, new int[] { 3, 3 }, 48, Activation.Relu);
            Function pool2 = layers.MaxPooling(conv2, new int[] { 3, 3 }, new int[] { 2, 2 });
            Function conv3 = layers.Convolution2D(pool2, new int[] { 3, 3 }, 64, Activation.Relu);
            Function f4    = layers.Dense(conv3, 96, Activation.Relu);
            Function drop4 = layers.Dropout(f4, 0.5);
            Function z     = layers.Dense(drop4, numOutputClasses);

            // Define loss and error metric.
            Function loss = CNTKLib.CrossEntropyWithSoftmax(z, labelVar);
            Function errorMetric = CNTKLib.ClassificationError(z, labelVar);

            // Training config.
            var minibatchSize = 64;
            var minibatchIterations = 200;

            // Instantiate progress writers.
            var trainingProgressOutputFreq = 100;

            // Instantiate the trainer object to drive the model training.
            var lrSchedule = new TrainingParameterScheduleDouble(0.01, 1);
            var learner = new List<Learner>() { Learner.SGDLearner(z.Parameters(), lrSchedule) };
            var trainer = Trainer.CreateTrainer(z, loss, errorMetric, learner);

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
                (var trainingImages, var trainingTargets) = readerTrain.NextBatch(minibatchSize);

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
                    if (samplesSeenNextBatch >= numberOftrainingSamples)
                    {
                        readerTrain.Reset();
                    }
                }
            }

            // Test model.
            var csharpError = MnistSimpleExampleTest.TestModel_Mnist_Loader(z, device, mnist);
            Trace.WriteLine($"Test Error: {csharpError}");

            // Save model.
            var modelPath = "cnn_mnist_csharp_loader.dnn";
            z.Save(modelPath);

            // Test loaded model.
            var loadedModel = Function.Load(modelPath, device);
            var loadedModelError = MnistSimpleExampleTest.TestModel_Mnist_Loader(loadedModel, device, mnist);

            Trace.WriteLine("Loaded Model Error: " + loadedModelError);
            Assert.AreEqual(csharpError, loadedModelError, 0.00001);

            // Test against python example.
            var pythonError = 0.89;
            Assert.AreEqual(pythonError, csharpError, 0.00001);
        }

        const string FeatureId = "features";
        const string LabelsId = "labels";

        [TestMethod]
        public void MnistCnnTest_MinibatchSource()
        {
            // Set global data and device type. 
            var dataType = CntkDataType.Float;
            DeviceDescriptor device = DeviceDescriptor.UseDefaultDevice();
            Trace.WriteLine("Using device: " + device.Type + " with data type: " + dataType);

            // Define data.
            var dataDirectoryPath = @"..\..\..\python\src\CntkPython\";
            var imageHeight = 28;
            var imageWidth = 28;
            var numChannels = 1;
            var numOutputClasses = 10;
            var inputDimensions = new int[] { imageWidth, imageHeight, numChannels };

            // Input variables denoting the features and label data.
            Variable inputVar = Variable.InputVariable(inputDimensions, dataType); ;
            Variable labelVar = Variable.InputVariable(new int[] { numOutputClasses }, dataType);

            // Instantiate the feedforward classification model
            var scaledInput = CNTKLib.ElementTimes(Constant.Scalar(dataType, 0.00390625, device), inputVar);

            // setup initializer
            var init = CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, seed: 32);

            var layers = new CntkLayers(device, dataType);
            //Function conv1 = layers.Convolution2D(scaledInput, new int[] { 5, 5 }, 32, (v) => CNTKLib.ReLU(v), init: init, bias: false, pad: true);
            //Function pool1 = layers.MaxPooling(conv1, new int[] { 3, 3 }, new int[] { 2, 2 });
            //Function conv2 = layers.Convolution2D(pool1, new int[] { 3, 3 }, 48, (v) => CNTKLib.ReLU(v), init: init, bias: false);
            //Function pool2 = layers.MaxPooling(conv2, new int[] { 3, 3 }, new int[] { 2, 2 });
            //Function conv3 = layers.Convolution2D(scaledInput, new int[] { 3, 3 }, 64, (v) => CNTKLib.ReLU(v), init: init, bias: false);
            Function f4 = layers.Dense(scaledInput, 96, Activation.Relu, init: init, bias: false);
            Function drop4 = layers.Dropout(f4, 0.5, seed: 32);
            Function z = layers.Dense(drop4, numOutputClasses, init: init, bias: false);

            // Define loss and error metric.
            Function loss = CNTKLib.CrossEntropyWithSoftmax(z, labelVar);
            Function errorMetric = CNTKLib.ClassificationError(z, labelVar);

            // Training config.
            uint minibatchSize = 64;
            var minibatchIterations = 200;

            // Instantiate progress writers.
            var trainingProgressOutputFreq = 100;

            // Instantiate the trainer object to drive the model training.
            var lrSchedule = new TrainingParameterScheduleDouble(0.01, 1);
            var learner = new List<Learner>() { Learner.SGDLearner(z.Parameters(), lrSchedule) };
            var trainer = Trainer.CreateTrainer(z, loss, errorMetric, learner);

            // Load train data.
            var trainPath = Path.Combine(dataDirectoryPath, "Train-28x28_cntk_text.txt");
            if (!File.Exists(trainPath))
            { throw new FileNotFoundException("Data not present. Use MNIST CNTK python example to download data"); }

            var readerTrain = CreateReader(trainPath,
                epochSize: 60000,
                inputDimensions: inputDimensions.Aggregate((d1, d2) => d1 * d2),
                numberOfClasses: numOutputClasses,
                randomize: false);

            var featureStreamInfo = readerTrain.StreamInfo(FeatureId);
            var labelStreamInfo = readerTrain.StreamInfo(LabelsId);

            // Train model.
            for (int i = 0; i < minibatchIterations; i++)
            {
                var mb = readerTrain.GetNextMinibatch(minibatchSize, device);
                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { inputVar, mb[featureStreamInfo] },
                    { labelVar, mb[labelStreamInfo] }
                };
                trainer.TrainMinibatch(arguments, device);

                if (((i + 1) % trainingProgressOutputFreq) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
                {
                    var trainLossValue = (float)trainer.PreviousMinibatchLossAverage();
                    var evaluationValue = (float)trainer.PreviousMinibatchEvaluationAverage();
                    Trace.WriteLine($"Minibatch: {i + 1} CrossEntropyLoss = {trainLossValue}, EvaluationCriterion = {evaluationValue}");
                }
            }

            // Test model.
            var mnist = Mnist.Load(DownloadPath);

            var csharpError = MnistSimpleExampleTest.TestModel_Mnist_Loader(z, device, mnist);
            Trace.WriteLine($"Test Error: {csharpError}");

            // Save model.
            var modelPath = "lr_mnist_csharp_loader.dnn";
            z.Save(modelPath);

            // Test loaded model.
            var loadedModel = Function.Load(modelPath, device);
            var loadedModelError = MnistSimpleExampleTest.TestModel_Mnist_Loader(z, device, mnist);

            Trace.WriteLine("Loaded Model Error: " + loadedModelError);
            Assert.AreEqual(csharpError, loadedModelError, 0.00001);

            // Test against python example.
            var pythonError = 0.202800;
            Assert.AreEqual(pythonError, csharpError, 0.00001);
        }

        MinibatchSource CreateReader(string path, ulong epochSize,
            int inputDimensions, int numberOfClasses, bool randomize)
        {
            var streamConfigurations = new StreamConfiguration[]
            {
                new StreamConfiguration(FeatureId, inputDimensions),
                new StreamConfiguration(LabelsId, numberOfClasses)
            };

            return MinibatchSource.TextFormatMinibatchSource(path, streamConfigurations, epochSize, randomize);
        }
    }
}
