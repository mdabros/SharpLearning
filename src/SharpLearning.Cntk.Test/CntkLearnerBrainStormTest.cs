using Microsoft.VisualStudio.TestTools.UnitTesting;
using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Cntk.Test
{
    [TestClass]
    public class CntkLearnerBrainStormTest
    {
        [TestMethod]
        public void TestMethod1()
        {
            var numberOfObservations = 500;
            var numberOfFeatures = 5;

            var random = new Random(32);

            var inputShape = new int[] { 5, 1 };
            var observationsData = Enumerable.Range(0, numberOfObservations * numberOfFeatures)
                .Select(v => (float)random.NextDouble()).ToArray();
            var observations = new MemoryMinibatchData<float>(observationsData, inputShape, numberOfObservations);

            var targetsData = Enumerable.Range(0, numberOfObservations)
                .Select(v => (float)random.NextDouble()).ToArray();
            var targets = new MemoryMinibatchData<float>(targetsData, new int[] { 1 }, numberOfObservations);

            var d = DataType.Float;
            var device = DeviceDescriptor.UseDefaultDevice();

            var network = Layers.Input(inputShape, d)
                .Dense(10, d, device)
                .ReLU()
                .Dense(1, d, device);

            var learner = new CntkLearner(network,
                p => Optimizers.SGD(p),
                (p, t) => Losses.MeanSquaredError(p, t),
                (p, t) => Losses.MeanAbsoluteError(p, t),
                d, device);

            var source = new MemoryMinibatchSource<float>(observations, targets, seed: 432, randomize: true);

            var model = learner.LearnFromSource(source, 32, 10);
        }

        CNTK.MinibatchSource CreateMinibatchSource(string map_file, int num_classes, bool train)
        {
            var transforms = new List<CNTK.CNTKDictionary>();
            if (true)
            {
                var randomSideTransform = CNTK.CNTKLib.ReaderCrop("RandomSide",
                  new Tuple<int, int>(0, 0),
                  new Tuple<float, float>(0.8f, 1.0f),
                  new Tuple<float, float>(0.0f, 0.0f),
                  new Tuple<float, float>(1.0f, 1.0f),
                  "uniRatio");
                transforms.Add(randomSideTransform);
            }

            var scaleTransform = CNTK.CNTKLib.ReaderScale(10, 10, 3);
            transforms.Add(scaleTransform);

            var imageDeserializer = CNTK.CNTKLib.ImageDeserializer(map_file, "labels", (uint)num_classes, "features", transforms);
            var minibatchSourceConfig = new CNTK.MinibatchSourceConfig(new CNTK.DictionaryVector() { imageDeserializer });
            return CNTK.CNTKLib.CreateCompositeMinibatchSource(minibatchSourceConfig);
        }
    }
}
