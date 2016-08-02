using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.FeatureTransformations.MatrixTransforms;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
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
    public class ClassificationMomentumNeuralNetLearnerTest
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

            var sut = new ClassificationMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.1f, 20, 0, 0.0, LearningRateSchedule.Constant);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.23831775700934579, actual);
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

            var sut = new ClassificationMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.1f, 20, 0.1, 0.0, LearningRateSchedule.Constant);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.397196261682243, actual);
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

            var sut = new ClassificationMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50, 0.5) }, new ReluActivation(), new LogLoss(),
                100, 0.1f, 20, 0.0, 0.0, LearningRateSchedule.Constant);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.40654205607476634, actual);
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

            var sut = new ClassificationMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.1f, 20, 0.0, 0.1, LearningRateSchedule.Constant);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.43925233644859812, actual);
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

            var sut = new ClassificationMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50, 0.5) }, new ReluActivation(), new LogLoss(),
                100, 0.1f, 20, 0.1, 0.1, LearningRateSchedule.Constant);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.4719626168224299, actual);
        }

        [TestMethod]
        public void ClassificationMomentumNeuralNetLearner_Learn_probabilities()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.1f, 20, 0, 0.0, LearningRateSchedule.Constant);
            var model = sut.Learn(observations, targets);

            var predictions = model.PredictProbability(observations);

            var evaluator = new LogLossClassificationProbabilityMetric();
            var actual = evaluator.Error(targets, predictions);                      

            Assert.AreEqual(0.73692409238739565, actual, 1e-6);
            Assert.AreEqual(0.23831775700934579, new TotalErrorClassificationMetric<double>().Error(targets, predictions.Select(p => p.Prediction).ToArray()), 1e-6);
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
            var evaluator = new LogLossClassificationProbabilityMetric();

            var sut = new ClassificationMomentumNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.1f, 20, 0, 0.0, LearningRateSchedule.Constant);

            var model = sut.LearnWithEarlyStopping(split.TrainingSet.Observations, split.TrainingSet.Targets,
                split.TestSet.Observations, split.TestSet.Targets, evaluator, 10);

            var predictions = model.PredictProbability(split.TestSet.Observations);

            var actualError = evaluator.Error(split.TestSet.Targets, predictions);
            Assert.AreEqual(0.71987659000502224, actualError, 1e-6);
            var actualIterations = model.Iterations;
            Assert.AreEqual(70, actualIterations);
        }
    }
}
