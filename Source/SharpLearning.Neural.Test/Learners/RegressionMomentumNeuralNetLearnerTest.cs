using Microsoft.VisualStudio.TestTools.UnitTesting;
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

            Assert.AreEqual(1.0039261698226392, actual, 0.00001);
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

            Assert.AreEqual(1.0479123167563083, actual, 0.00001);
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

            Assert.AreEqual(1.2586607075625156, actual, 0.00001);
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

            Assert.AreEqual(1.1444705821910322, actual, 0.00001);
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

            Assert.AreEqual(1.2661213154984761, actual, 0.00001);
        }
    }
}
