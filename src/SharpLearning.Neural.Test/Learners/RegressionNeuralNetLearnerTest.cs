using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Metrics.Regression;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;

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

            var (observations, targets) = CreateData(numberOfObservations, 
                numberOfFeatures, random);

            var net = new NeuralNet();
            net.Add(new InputLayer(numberOfFeatures));
            net.Add(new DenseLayer(10));
            net.Add(new SquaredErrorRegressionLayer());

            var sut = new RegressionNeuralNetLearner(net, new SquareLoss());
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.0871547675024143, actual, 0.0001);
        }

        [TestMethod]
        public void RegressionNeuralNetLearner_Learn_Early_Stopping()
        {
            var numberOfObservations = 500;
            var numberOfFeatures = 5;

            var random = new Random(32);

            var (observations, targets) = CreateData(numberOfObservations, 
                numberOfFeatures, random);

            var (validationObservations, validationTargets) = CreateData(numberOfObservations, 
                numberOfFeatures, random);

            var net = new NeuralNet();
            net.Add(new InputLayer(numberOfFeatures));
            net.Add(new DenseLayer(10));
            net.Add(new SquaredErrorRegressionLayer());

            var sut = new RegressionNeuralNetLearner(net, new SquareLoss(), 0.01, 150);
            var model = sut.Learn(observations, targets,
                validationObservations, validationTargets);

            var validationPredictions = model.Predict(validationObservations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(validationTargets, validationPredictions);

            Assert.AreEqual(0.093500629562319859, actual, 0.0001);
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

        (F64Matrix observations, double[] targets) CreateData(int numberOfObservations, int numberOfFeatures, Random random)
        {
            var observations = new F64Matrix(numberOfObservations, numberOfFeatures);
            observations.Map(() => random.NextDouble());
            var targets = Enumerable.Range(0, numberOfObservations).Select(i => random.NextDouble()).ToArray();

            return (observations, targets);
        }
    }
}
