using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Metrics.Regression;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;
using System;
using System.Linq;

namespace SharpLearning.Neural.Test.Learners
{
    [TestClass]
    public class RegressionNeuralNetLearnerTest
    {
        [TestMethod]
        public void RegressionNeuralNetLearner_Learn()
        {
            var numberOfObservations = 500;
            var numberOfFeatures = 5;

            var random = new Random(32);
            var observations = new F64Matrix(numberOfObservations, numberOfFeatures);
            observations.Initialize(() => random.NextDouble());
            var targets = Enumerable.Range(0, numberOfObservations).Select(i => random.NextDouble()).ToArray();

            var net = new NeuralNet();
            net.Add(new InputLayer(numberOfFeatures));
            net.Add(new DenseLayer(10));
            net.Add(new SquaredErrorRegressionLayer());

            var sut = new RegressionNeuralNetLearner(net, new SquareLoss());
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.086988999595858624, actual, 0.0001);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionNeuralNetLearner_Constructor_Throw_On_Wrong_OutputLayerType()
        {
            var net = new NeuralNet();
            net.Add(new InputLayer(10));
            net.Add(new DenseLayer(10));
            net.Add(new SvmLayer(10));

            var sut = new RegressionNeuralNetLearner(net, new AccuracyLoss());
        }
    }
}
