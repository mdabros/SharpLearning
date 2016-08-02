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
    public class RegressionMomentumNeuralNetLearnerTest
    {
        [TestMethod]
        public void ClassificationMomentumNeuralNetLearner_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new SquaredLoss(),
                100, 0.1f, 20, 0, 0.0, LearningRateSchedule.InvScaling);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(1.4383101999692078, actual, 0.00001);
        }

        [TestMethod]
        public void ClassificationMomentumNeuralNetLearner_Learn_L2Reguralization()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new SquaredLoss(),
                100, 0.1f, 20, 0.1, 0.0, LearningRateSchedule.InvScaling);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(1.4344494318970409, actual, 0.00001);
        }

        [TestMethod]
        public void ClassificationMomentumNeuralNetLearner_Learn_Hidden_Dropout()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50, 0.5) }, new ReluActivation(), new SquaredLoss(),
                100, 0.1f, 20, 0.0, 0.0, LearningRateSchedule.InvScaling);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(1.2596566860129561, actual, 0.00001);
        }

        [TestMethod]
        public void ClassificationMomentumNeuralNetLearner_Learn_Input_Dropout()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new SquaredLoss(),
                100, 0.1f, 20, 0.0, 0.1, LearningRateSchedule.InvScaling);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(1.2974122244570778, actual, 0.00001);
        }

        [TestMethod]
        public void ClassificationMomentumNeuralNetLearner_Learn_Input_Dropout_Hidden_Dropout_L2Reguralization()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50, 0.5) }, new ReluActivation(), new SquaredLoss(),
                100, 0.1f, 20, 0.1, 0.1, LearningRateSchedule.InvScaling);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(1.6727303009622458, actual, 0.00001);
        }

        [TestMethod]
        public void ClassificationMomentumNeuralNetLearner_Learn_Probabilities_WithEarlyStopping()
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

            var sut = new RegressionMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50, 0.5) }, new ReluActivation(), new SquaredLoss(),
                200, 0.1f, 20, 0.1, 0.1, LearningRateSchedule.InvScaling);

            var model = sut.LearnWithEarlyStopping(split.TrainingSet.Observations, split.TrainingSet.Targets,
                split.TestSet.Observations, split.TestSet.Targets, evaluator, 10);

            var predictions = model.Predict(split.TestSet.Observations);

            var actualError = evaluator.Error(split.TestSet.Targets, predictions);
            Assert.AreEqual(1.4622647541728322, actualError, 1e-6);
            var actualIterations = model.Iterations;
            Assert.AreEqual(10, actualIterations);
        }
    }
}
