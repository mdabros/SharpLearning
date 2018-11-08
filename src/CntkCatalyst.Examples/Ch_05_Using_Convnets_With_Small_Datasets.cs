using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using CNTK;
using CntkCatalyst.LayerFunctions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkCatalyst.Examples
{
    /// <summary>
    /// Example from Chapter 5.2: Using convnets with small datasets:
    /// https://github.com/mdabros/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb
    /// 
    /// This example needs manual download of the "dogs-vs-cats" dataset.
    /// Sources to download from:
    /// https://www.kaggle.com/c/dogs-vs-cats/data (needs an account)
    /// https://www.microsoft.com/en-us/download/details.aspx?id=54765
    /// </summary>
    [TestClass]
    public class Ch_05_Using_Convnets_With_Small_Datasets
    {
        [TestMethod]
        public void Run()
        {
            // Prepare data
            var baseDataDirectoryPath = @"E:\DataSets\CatsAndDogs";
            var mapFiles = PrepareMapFiles(baseDataDirectoryPath);

            // Define the input and output shape.
            var inputShape = new int[] { 150, 150, 3 };
            var numberOfClasses = 2;
            var outputShape = new int[] { numberOfClasses };

            // Setup minibatch sources.
            var featuresName = "features";
            var targetsName = "targets";

            var train = CreateMinibatchSource(mapFiles.trainFilePath, featuresName, targetsName,
                numberOfClasses, inputShape, augmentation: true);
            var trainingSource = new CntkMinibatchSource(train, featuresName, targetsName);

            // Notice augmentation is switched off for validation data.
            var valid = CreateMinibatchSource(mapFiles.validFilePath, featuresName, targetsName,
                numberOfClasses, inputShape, augmentation: false); 
            var validationSource = new CntkMinibatchSource(valid, featuresName, targetsName);

            // Notice augmentation is switched off for test data.
            var test = CreateMinibatchSource(mapFiles.testFilePath, featuresName, targetsName,
                numberOfClasses, inputShape, augmentation: false); 
            var testSource = new CntkMinibatchSource(test, featuresName, targetsName);

            // Define data type and device for the model.
            var dataType = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();
            
            // Create the architecture.
            var network = Layers.Input(inputShape, dataType)
                .Conv2D((3, 3), 32, (1, 1), device, dataType)
                .ReLU()
                .MaxPool2D((2, 2), (2, 2))

                .Conv2D((3, 3), 64, (1, 1), device, dataType)
                .ReLU()
                .MaxPool2D((2, 2), (2, 2))

                .Conv2D((3, 3), 128, (1, 1), device, dataType)
                .ReLU()
                .MaxPool2D((2, 2), (2, 2))

                .Conv2D((3, 3), 128, (1, 1), device, dataType)
                .ReLU()
                .MaxPool2D((2, 2), (2, 2))

                .Dense(512, device, dataType)
                .ReLU()
                .Dense(numberOfClasses, device, dataType)
                .Softmax();

            // Create the network.
            var model = new Sequential(network, dataType, device);

            // Compile the network with the selected learner, loss and metric.
            model.Compile(p => Learners.Adam(p, learningRate: 0.0001),
               (p, t) => Losses.CategoricalCrossEntropy(p, t),
               (p, t) => Metrics.Accuracy(p, t));

            // Write model summary.
            Trace.WriteLine(model.Summary());

            // Train the model using the training set.
            model.Fit(trainMinibatchSource: trainingSource,
                epochs: 100, batchSize: 32,
                validationMinibatchSource: validationSource);

            // Evaluate the model using the test set.
            (var loss, var metric) = model.Evaluate(testSource);

            // Write the test set loss and metric to debug output.
            Trace.WriteLine($"Test set - Loss: {loss}, Metric: {metric}");

            // Save model.
            model.Network.Save("cats_and_dogs_small_2.cntk");
        }

        MinibatchSource CreateMinibatchSource(string mapFilePath, string featuresName, string targetsName,
            int numberOfClasses, int[] inputShape, bool augmentation)
        {
            var transforms = new List<CNTKDictionary>();
            if (augmentation)
            {
                var randomSideTransform = CNTKLib.ReaderCrop(
                    cropType:"RandomSide",
                    cropSize: new Tuple<int, int>(0, 0),
                    sideRatio: new Tuple<float, float>(0.8f, 1.0f),
                    areaRatio: new Tuple<float, float>(0.0f, 0.0f),
                    aspectRatio: new Tuple<float, float>(1.0f, 1.0f),
                    jitterType: "uniRatio");

                transforms.Add(randomSideTransform);
            }

            var scaleTransform = CNTKLib.ReaderScale(inputShape[0], inputShape[1], inputShape[2]);
            transforms.Add(scaleTransform);

            var imageDeserializer = CNTKLib.ImageDeserializer(mapFilePath, targetsName, 
                (uint)numberOfClasses, featuresName, transforms);

            var minibatchSourceConfig = new MinibatchSourceConfig(new DictionaryVector() { imageDeserializer });
            return CNTKLib.CreateCompositeMinibatchSource(minibatchSourceConfig);
        }

        public static (string trainFilePath, string validFilePath, string testFilePath) PrepareMapFiles(
            string baseDataDirectoryPath)
        {
            var imageDirectoryPath = Path.Combine(baseDataDirectoryPath, "train");

            // Download data from one of these locations:
            // https://www.kaggle.com/c/dogs-vs-cats/data (needs an account)
            // https://www.microsoft.com/en-us/download/details.aspx?id=54765
            if (!Directory.Exists(imageDirectoryPath))
            {
                throw new ArgumentException($"Image data directory not found: {imageDirectoryPath}");
            }

            const int trainingSetSize = 1000;
            const int validationSetSize = 500;
            const int testSetSize = 500;

            const string trainFileName = "train_map.txt";
            const string validFileName = "validation_map.txt";
            const string testFileName = "test_map.txt";

            var fileNames = new string[] { trainFileName, validFileName, testFileName };
            var numberOfSamples = new int[] { trainingSetSize, validationSetSize, testSetSize };
            var counter = 0;

            for (int j = 0; j < fileNames.Length; j++)
            {
                var filename = fileNames[j];
                using (var distinationFileWriter = new System.IO.StreamWriter(filename, false))
                {
                    for (int i = 0; i < numberOfSamples[j]; i++)
                    {
                        var catFilePath = Path.Combine(imageDirectoryPath, "cat", $"cat.{counter}.jpg");
                        var dogFilePath = Path.Combine(imageDirectoryPath, "dog", $"dog.{counter}.jpg");
                        counter++;

                        distinationFileWriter.WriteLine($"{catFilePath}\t0");
                        distinationFileWriter.WriteLine($"{dogFilePath}\t1");
                    }
                }
                Trace.WriteLine("Wrote " + Path.Combine(Directory.GetCurrentDirectory(), filename));
            }

            return (trainFileName, validFileName, testFileName);
        }
    }
}
