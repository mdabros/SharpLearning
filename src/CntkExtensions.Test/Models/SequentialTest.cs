using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CntkExtensions.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Containers.Matrices;
using System.Diagnostics;

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
            //LoadMnistCsv(inputShape, outputShape);

            var network = new Sequential(Layers.Input(inputShape));

            network.Add(x => Layers.Dense(x, units: 512));
            network.Add(x => Layers.ReLU(x));
            network.Add(x => Layers.Dense(x, units: numberOfClasses));
            network.Add(x => Layers.Softmax(x));

            network.Compile(p => Learners.MomentumSGD(p),
               (t, p) => Losses.CategoricalCrossEntropy(t, p),
               (t, p) => Metrics.Accuracy(t, p));

            network.Fit(observations, targets, batchSize: 32, epochs: 100);

            (var loss, var metric) = network.Evaluate(observations, targets);

            Trace.WriteLine($"Final evaluation - Loss: {loss}, Metric: {metric}");
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
            targetsData = targetsData.Select(d => (float)random.Next(outputShape.Single())).ToArray();
            var oneHotTargetsData = targetsData.EncodeOneHot();

            var targetsShape = new List<int>(outputShape);
            targetsShape.Add(observationCount);
            var targets = new Tensor(oneHotTargetsData, targetsShape.ToArray());

            return (observations, targets);
        }

        static (Tensor observations, Tensor targets) LoadMnistCsv(int[] inputShape, int[] outputShape)
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
            var observationsShape = new List<int>(inputShape);
            observationsShape.Add(observationCount);
            var observations = new Tensor(trainingObservations.Data().Select(d => (float)d).ToArray(), 
                observationsShape.ToArray());

            // read classification targets (training)
            var trainingTargets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // Convert targest to float and one-hot
            var oneHotTargetsData = trainingTargets.Select(d => (float)d)
                .ToArray().EncodeOneHot();

            // convert targest tensor
            var targetsShape = new List<int>(outputShape);
            targetsShape.Add(observationCount);
            var targets = new Tensor(oneHotTargetsData, targetsShape.ToArray());

            return (observations, targets);
        }
    }
}
