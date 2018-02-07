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
    public class MnistSimpleExampleTest
    {
        const string DownloadPath = "MnistTest";

        [TestMethod]
        public void MnistSimpleTest()
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
                (var trainingImages, var trainingTargets) = readerTrain.NextBatch(minibatchSize);

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
            var readerTest = mnist.GetTestReader();

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
                (var testImages, var testTargets) = readerTest.NextBatch(minibatchSize);

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

            var pythonError = 0.186500; // Most likely from diffent observations, both in training and in test.
            Assert.AreEqual(pythonError, csharpError, 0.00001);

            // Save model.
            var modelPath = "lr_mnist_csharp_loader.dnn";
            model.Save(modelPath);

            var loadedModelError = LoadAndTestModel_Mnist_Loader(modelPath, device, mnist);

            Trace.WriteLine("Loaded Model Error: " + loadedModelError);

            Assert.AreEqual(csharpError, loadedModelError, 0.00001);
        }

        double LoadAndTestModel_Mnist_Loader(string modelPath, DeviceDescriptor device, Mnist mnist)
        {
            var loadedModel = Function.Load(modelPath, device);

            // Test loaded model.
            var numberOfTestSamples = mnist.TestImages.Count();
            var readerTest = mnist.GetTestReader(); // renew test reader.

            var inputVar = loadedModel.Arguments.Single();
            var inputShape = inputVar.Shape;
            var inputMap = new Dictionary<Variable, Value>();

            var outputVar = loadedModel.Output;
            var outputMap = new Dictionary<Variable, Value>();

            int errorCount = 0;
            for (int i = 0; i < numberOfTestSamples; i++)
            {
                (var testImages, var testTargets) = readerTest.NextBatch(1); // 1 example at a time.

                using (Value batchImages = Value.CreateBatch<float>(inputShape, testImages, device))
                {
                    inputMap.Add(inputVar, batchImages);
                    outputMap.Add(outputVar, null);

                    // Start eveluation on device.
                    loadedModel.Evaluate(inputMap, outputMap, device);

                    // Get evaluate result as dense output
                    var outputVal = outputMap[outputVar];
                    var outputData = outputVal.GetDenseData<float>(outputVar).Single(); // expect only one prediction with batch size 1.

                    var correctIndex = testTargets.ToList().IndexOf(1);
                    var max = outputData.Max();
                    var predictedIndex = outputData.IndexOf(max);
                    if (correctIndex != predictedIndex)
                    {
                        errorCount++;
                    }

                    inputMap.Clear();
                    outputMap.Clear();
                }
            }

            return errorCount / numberOfTestSamples;
        }

        const string FeatureId = "features";
        const string LabelsId = "labels";

        [TestMethod]
        public void MnistSimpleTest_MinibatchSource()
        {
            // Set global data and device type. 
            var dataType = CntkDataType.Float;
            DeviceDescriptor device = DeviceDescriptor.UseDefaultDevice();
            Trace.WriteLine("Using device: " + device.Type + " with data type: " + dataType);

            // Define data.
            var dataDirectoryPath = @"..\..\..\python\src\CntkPython\";
            var inputDimensions = 784;
            var numberOfClasses = 10;

            Trace.Write(Directory.GetCurrentDirectory());

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
            uint minibatchSize = 64;
            var minibatchIterations = 200;

            // Instantiate progress writers.
            var trainingProgressOutputFreq = 100;   

            // Instantiate the trainer object to drive the model training.
            var lr = new TrainingParameterScheduleDouble(0.01, 1);
            Trainer trainer = Trainer.CreateTrainer(model, loss, errorMetric,
                new List<Learner>() { Learner.SGDLearner(model.Parameters(), lr) });

            // Load train data.
            var trainPath = Path.Combine(dataDirectoryPath, "Train-28x28_cntk_text.txt");
            if (!File.Exists(trainPath))
            { throw new FileNotFoundException("Data not present. Use MNIST CNTK python example to download data"); }

            var readerTrain = CreateReader(trainPath,
                epochSize: 60000,
                inputDimensions: inputDimensions,
                numberOfClasses: numberOfClasses,
                randomize: false);

            var featureStreamInfo = readerTrain.StreamInfo(FeatureId);
            var labelStreamInfo = readerTrain.StreamInfo(LabelsId);

            // Train model.
            for (int i = 0; i < minibatchIterations; i++)
            {
                var mb = readerTrain.GetNextMinibatch(minibatchSize, device);
                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { x, mb[featureStreamInfo] },
                    { y, mb[labelStreamInfo] }
                };
                trainer.TrainMinibatch(arguments, device);

                if ((i % trainingProgressOutputFreq) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
                {
                    var trainLossValue = (float)trainer.PreviousMinibatchLossAverage();
                    var evaluationValue = (float)trainer.PreviousMinibatchEvaluationAverage();
                    Trace.WriteLine($"Minibatch: {i} CrossEntropyLoss = {trainLossValue}, EvaluationCriterion = {evaluationValue}");
                }
            }

            // Load test data.
            var testPath = Path.Combine(dataDirectoryPath, "Test-28x28_cntk_text.txt");
            if (!File.Exists(testPath))
            { throw new FileNotFoundException("Data not present. Use MNIST CNTK python example to download data"); }

            var readerTest = CreateReader(testPath,
                epochSize: 10000,
                inputDimensions: inputDimensions,
                numberOfClasses: numberOfClasses,
                randomize: false);

            featureStreamInfo = readerTest.StreamInfo(FeatureId);
            labelStreamInfo = readerTest.StreamInfo(LabelsId);

            // Test data for trained model.
            var testMinibatchSize = 1024;
            var numberOfTestSamples = 10000;
            var numMinibatchesToTest = numberOfTestSamples / testMinibatchSize;
            var testResult = 0.0;

            for (int i = 0; i < numMinibatchesToTest; i++)
            {
                var mb = readerTest.GetNextMinibatch(minibatchSize);
                var arguments = new UnorderedMapVariableMinibatchData
                {
                    { x, mb[featureStreamInfo] },
                    { y, mb[labelStreamInfo] }
                };
                var evalError = trainer.TestMinibatch(arguments, device);

                testResult += evalError;
            }

            var csharpError = testResult / numMinibatchesToTest;
            Trace.WriteLine($"Test Error: {csharpError}");

            var pythonError = 0.186500; // Most likely from diffent observations during training.
            Assert.AreEqual(pythonError, csharpError, 0.00001);

            // Save model.
            var modelPath = "lr_minibatch_source.dnn";
            model.Save(modelPath);

            // Evaluate the C# Mnist test set. This should be changed to the minibatch source dataset.
            var loadedModelError = LoadAndTestModel_Mnist_Loader(modelPath, device, Mnist.Load(DownloadPath));

            Trace.WriteLine("Loaded Model Error: " + loadedModelError);

            Assert.AreEqual(csharpError, loadedModelError, 0.00001);
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
