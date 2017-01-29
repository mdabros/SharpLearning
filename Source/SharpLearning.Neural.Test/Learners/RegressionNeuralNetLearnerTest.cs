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

            F64Matrix observations;
            double[] targets;
            CreateData(numberOfObservations, numberOfFeatures, random, out observations, out targets);

            var net = new NeuralNet();
            net.Add(new InputLayer(numberOfFeatures));
            net.Add(new DenseLayer(10));
            net.Add(new SquaredErrorRegressionLayer());

            var sut = new RegressionNeuralNetLearner(net, new SquareLoss());
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.089005424175892453, actual, 0.0001);
        }

        [TestMethod]
        public void RegressionNeuralNetLearner_Learn_Early_Stopping()
        {
            var numberOfObservations = 500;
            var numberOfFeatures = 5;

            var random = new Random(32);

            F64Matrix observations;
            double[] targets;
            CreateData(numberOfObservations, numberOfFeatures, random, out observations, out targets);

            F64Matrix validationObservations;
            double[] validationTargets;
            CreateData(numberOfObservations, numberOfFeatures, random, out validationObservations, out validationTargets);

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

            Assert.AreEqual(0.093934776279484114, actual, 0.0001);
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

        void CreateData(int numberOfObservations, int numberOfFeatures, Random random, out F64Matrix observations, out double[] targets)
        {
            observations = new F64Matrix(numberOfObservations, numberOfFeatures);
            observations.Initialize(() => random.NextDouble());
            targets = Enumerable.Range(0, numberOfObservations).Select(i => random.NextDouble()).ToArray();
        }

    }
}
