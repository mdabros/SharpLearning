using System;
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
    }
}
