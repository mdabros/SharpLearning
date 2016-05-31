using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Ensemble.Learners;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.Ensemble.Test.Properties;
using SharpLearning.Metrics.Classification;
using System.Linq;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Ensemble.EnsembleSelectors;

namespace SharpLearning.Ensemble.Test.Learners
{
    [TestClass]
    public class ClassificationForwardSearchModelSelectingEnsembleLearnerTest
    {
        [TestMethod]
        public void ClassificationForwardSearchModelSelectingEnsembleLearner_Learn()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9),
                new ClassificationDecisionTreeLearner(11),
                new ClassificationDecisionTreeLearner(21),
                new ClassificationDecisionTreeLearner(23),
                new ClassificationDecisionTreeLearner(1),
                new ClassificationDecisionTreeLearner(14),
                new ClassificationDecisionTreeLearner(17),
                new ClassificationDecisionTreeLearner(19),
                new ClassificationDecisionTreeLearner(33)
            };

            var sut = new ClassificationForwardSearchModelSelectingEnsembleLearner(learners, 5);

            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var model = sut.Learn(observations, targets);
            var predictions = model.PredictProbability(observations);

            var metric = new LogLossClassificationProbabilityMetric();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.51787562976713208, actual, 0.0001);
        }

        [TestMethod]
        public void ClassificationForwardSearchModelSelectingEnsembleLearner_Learn_Without_Replacement()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9),
                new ClassificationDecisionTreeLearner(11),
                new ClassificationDecisionTreeLearner(21),
                new ClassificationDecisionTreeLearner(23),
                new ClassificationDecisionTreeLearner(1),
                new ClassificationDecisionTreeLearner(14),
                new ClassificationDecisionTreeLearner(17),
                new ClassificationDecisionTreeLearner(19),
                new ClassificationDecisionTreeLearner(33)
            };

            var metric = new LogLossClassificationProbabilityMetric();
            var ensembleStrategy = new MeanProbabilityClassificationEnsembleStrategy();

            var sut = new ClassificationForwardSearchModelSelectingEnsembleLearner(learners, 5,
                new StratifiedCrossValidation<ProbabilityPrediction>(5, 23), ensembleStrategy, metric, 1, false);

            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var model = sut.Learn(observations, targets);
            var predictions = model.PredictProbability(observations);

            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.53941649903795308, actual, 0.0001);
        }

        [TestMethod]
        public void ClassificationForwardSearchModelSelectingEnsembleLearner_Learn_Start_With_3_Models()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9),
                new ClassificationDecisionTreeLearner(11),
                new ClassificationDecisionTreeLearner(21),
                new ClassificationDecisionTreeLearner(23),
                new ClassificationDecisionTreeLearner(1),
                new ClassificationDecisionTreeLearner(14),
                new ClassificationDecisionTreeLearner(17),
                new ClassificationDecisionTreeLearner(19),
                new ClassificationDecisionTreeLearner(33)
            };

            var metric = new LogLossClassificationProbabilityMetric();
            var ensembleStrategy = new MeanProbabilityClassificationEnsembleStrategy();

            var sut = new ClassificationForwardSearchModelSelectingEnsembleLearner(learners, 5,
                new StratifiedCrossValidation<ProbabilityPrediction>(5, 23), ensembleStrategy, metric, 3, true);

            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var model = sut.Learn(observations, targets);
            var predictions = model.PredictProbability(observations);

            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.54434276244488244, actual, 0.0001);
        }

        [TestMethod]
        public void ClassificationForwardSearchModelSelectingEnsembleLearner_Learn_Indexed()
        {
            var learners = new IIndexedLearner<ProbabilityPrediction>[]
            {
                new ClassificationDecisionTreeLearner(2),
                new ClassificationDecisionTreeLearner(5),
                new ClassificationDecisionTreeLearner(7),
                new ClassificationDecisionTreeLearner(9),
                new ClassificationDecisionTreeLearner(11),
                new ClassificationDecisionTreeLearner(21),
                new ClassificationDecisionTreeLearner(23),
                new ClassificationDecisionTreeLearner(1),
                new ClassificationDecisionTreeLearner(14),
                new ClassificationDecisionTreeLearner(17),
                new ClassificationDecisionTreeLearner(19),
                new ClassificationDecisionTreeLearner(33)
            };

            var metric = new LogLossClassificationProbabilityMetric();
            var ensembleStrategy = new MeanProbabilityClassificationEnsembleStrategy();

            var sut = new ClassificationForwardSearchModelSelectingEnsembleLearner(learners, 5,
                new RandomCrossValidation<ProbabilityPrediction>(5, 23), ensembleStrategy, metric, 3, true);

            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var indices = Enumerable.Range(0, 25).ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.PredictProbability(observations);

            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(2.3682546920482164, actual, 0.0001);
        }
    }
}
