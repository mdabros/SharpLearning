using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Metrics.Classification;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;
using System;
using System.Linq;

namespace SharpLearning.Neural.Test.Learners
{
    [TestClass]
    public class ClassificationNeuralNetLearnerTest
    {
        [TestMethod]
        public void ClassificationNeuralNetLearner_Learn()
        {
            var numberOfObservations = 500;
            var numberOfFeatures = 5;
            var numberOfClasses = 5;

            var random = new Random(32);
            F64Matrix observations;
            double[] targets;
            CreateData(numberOfObservations, numberOfFeatures, numberOfClasses, random, out observations, out targets);

            var net = new NeuralNet();
            net.Add(new InputLayer(numberOfFeatures));
            net.Add(new DenseLayer(10));
            net.Add(new SvmLayer(numberOfClasses));

            var sut = new ClassificationNeuralNetLearner(net, new AccuracyLoss());
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.77, actual);
        }

        [TestMethod]
        public void ClassificationNeuralNetLearner_Learn_Early_Stopping()
        {
            var numberOfObservations = 500;
            var numberOfFeatures = 5;
            var numberOfClasses = 5;

            var random = new Random(32);

            F64Matrix observations;
            double[] targets;
            CreateData(numberOfObservations, numberOfFeatures, numberOfClasses, random, out observations, out targets);

            F64Matrix validationObservations;
            double[] validationTargets;
            CreateData(numberOfObservations, numberOfFeatures, numberOfClasses, random, out validationObservations, out validationTargets);

            var net = new NeuralNet();
            net.Add(new InputLayer(numberOfFeatures));
            net.Add(new DenseLayer(10));
            net.Add(new SvmLayer(numberOfClasses));

            var sut = new ClassificationNeuralNetLearner(net, new AccuracyLoss());
            var model = sut.Learn(observations, targets,
                validationObservations, validationTargets);

            var validationPredictions = model.Predict(validationObservations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(validationTargets, validationPredictions);

            Assert.AreEqual(0.802, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ClassificationNeuralNetLearner_Constructor_Throw_On_Wrong_OutputLayerType()
        {
            var net = new NeuralNet();
            net.Add(new InputLayer(10));
            net.Add(new DenseLayer(10));
            net.Add(new SquaredErrorRegressionLayer());

            var sut = new ClassificationNeuralNetLearner(net, new AccuracyLoss());
        }

        void CreateData(int numberOfObservations, int numberOfFeatures, int numberOfClasses, Random random, out F64Matrix observations, out double[] targets)
        {
            observations = new F64Matrix(numberOfObservations, numberOfFeatures);
            observations.Map(() => random.NextDouble());
            targets = Enumerable.Range(0, numberOfObservations).Select(i => (double)random.Next(0, numberOfClasses)).ToArray();
        }
    }
}
