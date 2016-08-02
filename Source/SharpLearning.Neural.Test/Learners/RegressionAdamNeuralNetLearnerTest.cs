using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.FeatureTransformations.MatrixTransforms;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.Test.Properties;
using System.IO;
using System.Linq;

namespace SharpLearning.Neural.Test.Learners
{
    [TestClass]
    public class RegressionAdamNeuralNetLearnerTest
    {
        [TestMethod]
        public void RegressionAdamNeuralNetLearner_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionAdamNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new SquaredLoss(),
                100, 0.001, 20, 0);

            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(1.27833029857311, actual, 0.00001);
        }

        [TestMethod]
        public void RegressionAdamNeuralNetLearner_Learn_L2Regularization()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionAdamNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new SquaredLoss(),
                100, 0.001, 20, 0.1);

            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(1.2667904877203977, actual, 0.00001);
        }

        [TestMethod]
        public void RegressionAdamNeuralNetLearner_Learn_Hidden_Dropout()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionAdamNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50, 0.5) }, new ReluActivation(), new SquaredLoss(),
                100, 0.001, 20, 0.0);

            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(1.4795178477497608, actual, 0.00001);
        }

        [TestMethod]
        public void RegressionAdamNeuralNetLearner_Learn_Input_Dropout()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionAdamNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new SquaredLoss(),
                100, 0.001, 20, 0.0, 0.1);

            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(1.4015343169522598, actual, 0.00001);
        }

        [TestMethod]
        public void RegressionAdamNeuralNetLearner_Learn_Input_Dropout_Hidden_Dropout_L2Reguralization()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionAdamNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50, 0.5) }, new ReluActivation(), new SquaredLoss(),
                100, 0.001, 20, 0.1, 0.1);

            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(2.665742824849064, actual, 0.00001);
        }

        [TestMethod]
        public void RegressionAdamNeuralNetLearner_Learn_Probabilities_WithEarlyStopping()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var splitter = new RandomTrainingTestIndexSplitter<double>(0.7, 21);
            var split = splitter.SplitSet(observations, targets);
            var evaluator = new MeanSquaredErrorRegressionMetric();

            var sut = new RegressionAdamNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50, 0.5) }, new ReluActivation(), new SquaredLoss(),
                200, 0.001, 20, 0.1, 0.1);

            var model = sut.LearnWithEarlyStopping(split.TrainingSet.Observations, split.TrainingSet.Targets,
                split.TestSet.Observations, split.TestSet.Targets, evaluator, 10);

            var predictions = model.Predict(split.TestSet.Observations);

            var actualError = evaluator.Error(split.TestSet.Targets, predictions);
            Assert.AreEqual(1.4824465345651972, actualError, 1e-6);
            var actualIterations = model.Iterations;
            Assert.AreEqual(100, actualIterations);
        }
    }
}
