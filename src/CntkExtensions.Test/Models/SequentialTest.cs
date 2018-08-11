using System;
using System.Collections.Generic;
using System.Linq;
using CntkExtensions.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CntkExtensions.Test.Models
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

            var network = new Sequential(Layers.Input(inputShape));

            network.Add(x => Layers.Dense(x, units: 512));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: numberOfClasses));
            network.Add(x => Layers.Softmax(x));

            network.Compile(p => Learners.MomentumSGD(p),
               (t, p) => Losses.CategoricalCrossEntropy(t, p),
               (t, p) => Metrics.Accuracy(t, p));

            network.Fit(observations, targets, batchSize: 32, epochs: 5);
        }

        static (Tensor observations, Tensor targets) CreateArtificialData(int[] inputShape, int[] outputShape, int observationCount)
        {
            var inputSize = inputShape.Aggregate((d1, d2) => d1 * d2);
            var random = new Random(32);

            var observationsData = new float[observationCount * inputSize];
            observationsData = observationsData.Select(v => (float)random.NextDouble()).ToArray();

            var observationsShape = new List<int>(inputShape);
            observationsShape.Add(observationCount);
            var observations = new Tensor(observationsData, observationsShape.ToArray());

            var targetsData = new float[observationCount];
            targetsData = targetsData.Select(d => (float)random.Next(10)).ToArray();
            var oneHotTargetsData = targetsData.EncodeOneHot();

            var targetsShape = new List<int>(outputShape);
            targetsShape.Add(observationCount);
            var targets = new Tensor(oneHotTargetsData, targetsShape.ToArray());

            return (observations, targets);
        }
    }
}
