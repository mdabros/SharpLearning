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
            var observations = new F64Matrix(numberOfObservations, numberOfFeatures);
            observations.Initialize(() => random.NextDouble());
            var targets = Enumerable.Range(0, numberOfObservations).Select(i => (double)random.Next(0, numberOfClasses)).ToArray();

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
        [ExpectedException(typeof(ArgumentException))]
        public void ClassificationNeuralNetLearner_Constructor_Throw_On_Wrong_OutputLayerType()
        {
            var net = new NeuralNet();
            net.Add(new InputLayer(10));
            net.Add(new DenseLayer(10));
            net.Add(new SquaredErrorRegressionLayer());

            var sut = new ClassificationNeuralNetLearner(net, new AccuracyLoss());
        }
    }
}
