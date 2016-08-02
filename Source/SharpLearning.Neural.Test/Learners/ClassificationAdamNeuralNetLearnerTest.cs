using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.FeatureTransformations;
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
    public class ClassificationAdamNeuralNetLearnerTest
    {
        [TestMethod]
        public void ClassificationAdamNeuralNetLearner_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
           
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationAdamNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.01, 20, 0);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.3364485981308411, actual);
        }

        [TestMethod]
        public void ClassificationAdamNeuralNetLearner_Learn_L2Reguralization()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();

            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationAdamNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.01, 20, 0.1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.38317757009345793, actual);
        }

        [TestMethod]
        public void ClassificationAdamNeuralNetLearner_Learn_Hidden_Dropout()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();

            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationAdamNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50, 0.5) }, new ReluActivation(), new LogLoss(),
                100, 0.01, 20, 0.0);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.43457943925233644, actual);
        }

        [TestMethod]
        public void ClassificationAdamNeuralNetLearner_Learn_Input_Dropout()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();

            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationAdamNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.01, 20, 0.0, 0.1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.42523364485981308, actual);
        }

        [TestMethod]
        public void ClassificationAdamNeuralNetLearner_Learn_Input_Dropout_Hidden_Dropout_L2Reguraliztion()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();

            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationAdamNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50, 0.5) }, new ReluActivation(), new LogLoss(),
                100, 0.01, 20, 0.1, 0.1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.46261682242990654, actual);
        }

        [TestMethod]
        public void ClassificationAdamNeuralNetLearner_Learn_probabilities()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var features = parser.EnumerateRows(v => v != "Target").First().ColumnNameToIndex.Keys.ToArray();
            var normalizer = new MinMaxTransformer(0.0, 1.0);
            var observations = parser.EnumerateRows(features)
                .ToF64Matrix();
            normalizer.Transform(observations, observations);

            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationAdamNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.01, 20, 0);

            var model = sut.Learn(observations, targets);

            var predictions = model.PredictProbability(observations);

            var evaluator = new LogLossClassificationProbabilityMetric();
            var actual = evaluator.Error(targets, predictions);
            
            Assert.AreEqual(0.87556824055160276, actual, 1e-6);
            Assert.AreEqual(0.3364485981308411, new TotalErrorClassificationMetric<double>().Error(targets, predictions.Select(p => p.Prediction).ToArray()), 1e-6);
        }

        [TestMethod]
        public void ClassificationAdamNeuralNetLearner_Learn_Probabilities_WithEarlyStopping()
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

            var sut = new ClassificationAdamNeuralNetLearner(new HiddenLayer[] { HiddenLayer.New(50) }, new ReluActivation(), new LogLoss(),
                100, 0.01, 20, 0);

            var model = sut.LearnWithEarlyStopping(split.TrainingSet.Observations, split.TrainingSet.Targets,
                split.TestSet.Observations, split.TestSet.Targets, evaluator, 10);

            var predictions = model.PredictProbability(split.TestSet.Observations);

            var actualError = evaluator.Error(split.TestSet.Targets, predictions);
            Assert.AreEqual(0.85893151684561364, actualError, 1e-6);
            var actualIterations = model.Iterations;
            Assert.AreEqual(30, actualIterations);
        }
    }
}
