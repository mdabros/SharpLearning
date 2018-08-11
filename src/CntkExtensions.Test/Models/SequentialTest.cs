using System;
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
            var observationCount = 10000;
            var inputShape = new int[] { 28, 28, 1 };
            var inputSize = inputShape.Aggregate((d1, d2) => d1 * d2);
            var outputShape = 10;

            var random = new Random(32);
            var observationsData = new float[observationCount * inputSize];
            observationsData = observationsData.Select(v => (float)random.NextDouble()).ToArray();
            var observations = new Tensor(observationsData, 28, 28, 1, observationCount);
            
            var targetsData = new float[observationCount];
            targetsData = targetsData.Select(d => (float)random.Next(10)).ToArray();
            var oneHotTargetsData = targetsData.EncodeOneHot();
            var targets = new Tensor(oneHotTargetsData, outputShape, observationCount);

            var network = new Sequential(Layers.Input(inputShape));

            network.Add(x => Layers.Dense(x, units: 512));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: outputShape));

            network.Compile(p => Learners.MomentumSGD(p), 
               (t, p) => Losses.CategoricalCrossEntropy(t, p), 
               (t, p) => Metrics.Accuracy(t, p));

            network.Fit(observations, targets, batchSize: 32, epochs: 5);


        }
    }
}
