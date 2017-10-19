using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CNTK;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Metrics.Classification;
using SharpLearning.Neural.Cntk;

namespace SharpLearning.Neural.Test.Cntk
{
    [TestClass]
    public class CntkNeuralNetLearnerTest
    {
        [TestMethod]
        public void CntkNeuralNetLearner_Learn()
        {
            var numberOfObservations = 500;
            var numberOfFeatures = 5;
            var numberOfClasses = 5;

            var random = new Random(32);
            F64Matrix observations;
            double[] targets;
            CreateData(numberOfObservations, numberOfFeatures, numberOfClasses, random, out observations, out targets);

            // set global device.
            CntkLayers.Device = DeviceDescriptor.CPUDevice;

            var net = CntkLayers.Input(numberOfFeatures);
            net = CntkLayers.Dense(net, 10);
            net = CntkLayers.Activation(net, Activation.ReLU);
            net = CntkLayers.Dense(net, numberOfClasses);

            var sut = new CntkNeuralNetLearner(net, CntkLayers.Device);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.79, actual);
        }

        void CreateData(int numberOfObservations, int numberOfFeatures, int numberOfClasses, Random random, out F64Matrix observations, out double[] targets)
        {
            observations = new F64Matrix(numberOfObservations, numberOfFeatures);
            observations.Map(() => random.NextDouble());
            targets = Enumerable.Range(0, numberOfObservations).Select(i => (double)random.Next(0, numberOfClasses)).ToArray();
        }

        //[TestMethod]
        public void CntkNeuralNetLearner_Learn_From_MinibatchSoruce()
        {
            var epochs = 10;
            var numberOfClasses = 10;
            var imageDimensions = new int[] { 32, 32, 3 };

            var net = CntkLayers.Input(imageDimensions);
            net = CntkLayers.Conv2D(net, 5, 5, 8);
            net = CntkLayers.Activation(net, Activation.ReLU);
            net = CntkLayers.Dense(net, 64);
            net = CntkLayers.Dropout(net, 0.2, 123);
            net = CntkLayers.Activation(net, Activation.ReLU);
            net = CntkLayers.Dense(net, numberOfClasses);
            var sut = new CntkNeuralNetLearner(net, CntkLayers.Device, 0.001, epochs, batchSize: 64);

            var cifarDirectory = @"K:\Git\CNTK\Examples\Image\DataSets\CIFAR-10\";

            var dataMapFile = Path.Combine(cifarDirectory, "train_map.txt");
            var meanFile = Path.Combine(cifarDirectory, "CIFAR-10_mean.xml");

            var minibatchSource = CreateMinibatchSource(dataMapFile, meanFile,
                imageDimensions, numberOfClasses, (uint)epochs);

            var model = sut.Learn(minibatchSource, "features", "labels", numberOfClasses);
        }

        static MinibatchSource CreateMinibatchSource(string mapFilePath, string meanFilePath,
            int[] imageDims, int numClasses, uint maxSweeps)
        {
            var transforms = new List<CNTKDictionary>{
                CNTKLib.ReaderCrop("RandomSide",
                    new Tuple<int, int>(0, 0),
                    new Tuple<float, float>(0.8f, 1.0f),
                    new Tuple<float, float>(0.0f, 0.0f),
                    new Tuple<float, float>(1.0f, 1.0f),
                    "uniRatio"),
                CNTKLib.ReaderScale(imageDims[0], imageDims[1], imageDims[2]),
                CNTKLib.ReaderMean(meanFilePath)
            };

            var deserializerConfiguration = CNTKLib.ImageDeserializer(mapFilePath,
                "labels", (uint)numClasses,
                "features",
                transforms);

            MinibatchSourceConfig config = new MinibatchSourceConfig(new List<CNTKDictionary> { deserializerConfiguration })
            {
                MaxSweeps = maxSweeps
            };

            return CNTKLib.CreateCompositeMinibatchSource(config);
        }
    }
}
