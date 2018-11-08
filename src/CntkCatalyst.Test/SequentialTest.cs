using System;
using System.Diagnostics;
using System.Linq;
using CNTK;
using CntkCatalyst.LayerFunctions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

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

            var dataType = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();

            // Create the architecture.
            var network = Layers.Input(inputShape, dataType)
                .Dense(512, device, dataType)
                .ReLU()
                .Dense(numberOfClasses, device, dataType)
                .Softmax();

            var model = new Sequential(network, dataType, device);

            model.Compile(p => Learners.MomentumSGD(p),
               (p, t) => Losses.CategoricalCrossEntropy(p, t),
               (p, t) => Metrics.Accuracy(p, t));

            var trainSource = new MemoryMinibatchSource(observations, targets, seed: 232, randomize: true);
            model.Fit(trainSource, batchSize: 32, epochs: 10);

            var evalSource = new MemoryMinibatchSource(observations, targets, seed: 232, randomize: false);
            (var loss, var metric) = model.Evaluate(trainSource);

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
    }
}
