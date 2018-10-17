using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using CNTK;
using CntkCatalyst.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;

namespace CntkCatalyst.Test.Models
{
    [TestClass]
    public class SequentialTest
    {
        [TestMethod]
        public void Sequential_Use_Case()
        {
            var inputShape = new int[] { 28, 28, 1 };
            var numberOfClasses = 10;
            var outputShape = new int[] { numberOfClasses };

            (var observations, var targets) = CreateArtificialData(inputShape, outputShape, observationCount: 10000);

            var d = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();
            var network = new Sequential(Layers.Input(inputShape), d, device);

            network.Add(x => Layers.Dense(x, units: 512));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: numberOfClasses));
            network.Add(x => Layers.Softmax(x));

            network.Compile(p => Learners.MomentumSGD(p),
               (p, t) => Losses.CategoricalCrossEntropy(p, t),
               (p, t) => Metrics.Accuracy(p, t));

            network.Fit(observations, targets, batchSize: 32, epochs: 100);

            (var loss, var metric) = network.Evaluate(observations, targets);

            Trace.WriteLine($"Final evaluation - Loss: {loss}, Metric: {metric}");
        }

        static (MemoryMinibatchData observations, MemoryMinibatchData targets) CreateArtificialData(int[] inputShape, int[] outputShape, int observationCount)
        {
            var inputSize = inputShape.Aggregate((d1, d2) => d1 * d2);
            var random = new Random(32);

            var observationsData = new float[observationCount * inputSize];
            observationsData = observationsData.Select(v => (float)random.NextDouble()).ToArray();

            var observations = new MemoryMinibatchData(observationsData, inputShape.ToArray(), observationCount);

            var targetsData = new float[observationCount];
            targetsData = targetsData.Select(d => (float)random.Next(outputShape.Single())).ToArray();
            var oneHotTargetsData = targetsData.EncodeOneHot();

            var targets = new MemoryMinibatchData(oneHotTargetsData, outputShape, observationCount);

            return (observations, targets);
        }

        static (MemoryMinibatchData observations, MemoryMinibatchData targets) LoadMnistCsv(int[] inputShape, int[] outputShape)
        {
            var parser = new CsvParser(() => new StreamReader(@"E:\Git\SharpLearning.Examples\src\Resources\mnist_small_train.csv"));
            var targetName = "Class";

            var featureNames = parser.EnumerateRows(c => c != targetName).First().ColumnNameToIndex.Keys.ToArray();

            // read feature matrix (training)
            var trainingObservations = parser
                .EnumerateRows(featureNames)
                .ToF64Matrix();

            // transform pixel values to be between 0 and 1.
            trainingObservations.Map(p => p / 255);

            var observationCount = trainingObservations.RowCount;
            
            // convert to float tensor
            var observations = new MemoryMinibatchData(trainingObservations.Data().Select(d => (float)d).ToArray(), 
                inputShape.ToArray(), observationCount);

            // read classification targets (training)
            var trainingTargets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // Convert targest to float and one-hot
            var oneHotTargetsData = trainingTargets.Select(d => (float)d)
                .ToArray().EncodeOneHot();

            // convert targest tensor
            var targets = new MemoryMinibatchData(oneHotTargetsData, outputShape.ToArray(), observationCount);

            return (observations, targets);
        }
    }
}
