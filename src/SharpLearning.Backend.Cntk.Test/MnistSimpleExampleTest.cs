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

                    if (((i + 1) % trainingProgressOutputFreq) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
                    {
                        var trainLossValue = trainer.PreviousMinibatchLossAverage();
                        var evaluationValue = trainer.PreviousMinibatchEvaluationAverage();
                        Trace.WriteLine($"Minibatch: {i + 1} CrossEntropyLoss = {trainLossValue:F16}, EvaluationCriterion = {evaluationValue:F16}");
                    }

                    var samplesSeenNextBatch = (i + 2) * minibatchSize;
                    if (samplesSeenNextBatch >= numberOftrainingSamples)
                    {
                        readerTrain.Reset();
                    }
                }
            }

            // Test model.
            var csharpError = TestModelMnistLoader(model, device, mnist);
            Trace.WriteLine($"Test Error: {csharpError}");

            // Save model.
            var modelPath = "lr_mnist_csharp_loader.dnn";
            model.Save(modelPath);

            // Test loaded model.
            var loadedModel = Function.Load(modelPath, device);
            var loadedModelError = TestModelMnistLoader(loadedModel, device, mnist);

            Trace.WriteLine("Loaded Model Error: " + loadedModelError);
            Assert.AreEqual(csharpError, loadedModelError, 0.00001);

            // Test against python example.
            var pythonError = 0.202800;
            Assert.AreEqual(pythonError, csharpError, 0.00001);
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

                if (((i + 1) % trainingProgressOutputFreq) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
                {
                    var trainLossValue = trainer.PreviousMinibatchLossAverage();
                    var evaluationValue = trainer.PreviousMinibatchEvaluationAverage();
                    Trace.WriteLine($"Minibatch: {i + 1} CrossEntropyLoss = {trainLossValue:F16}, EvaluationCriterion = {evaluationValue:F16}");
                }
            }

            // Test model.
            var mnist = Mnist.Load(DownloadPath);

            var csharpError = TestModelMnistLoader(model, device, mnist);
            Trace.WriteLine($"Test Error: {csharpError}");

            // Save model.
            var modelPath = "lr_mnist_csharp_loader.dnn";
            model.Save(modelPath);

            // Test loaded model.
            var loadedModel = Function.Load(modelPath, device);
            var loadedModelError = TestModelMnistLoader(loadedModel, device, mnist);

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

        public static double TestModelMnistLoader(Function model, DeviceDescriptor device, Mnist mnist)
        {
            var numberOfTestSamples = mnist.TestImages.Count();
            var readerTest = mnist.GetTestReader(); // renew test reader.

            var inputVar = model.Arguments.Single();
            var inputShape = inputVar.Shape;
            var inputMap = new Dictionary<Variable, Value>();

            var outputVar = model.Output;
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
                    model.Evaluate(inputMap, outputMap, device);

                    // Get evaluation result as dense output.
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

            return (double)errorCount / (double)numberOfTestSamples;
        }
    }
}
