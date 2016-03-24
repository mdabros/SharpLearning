using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Common.Interfaces;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Ensemble.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using System.IO;
using System.Linq;

namespace SharpLearning.Ensemble.Test.Learners
{
    [TestClass]
    public class RegressionStackingEnsembleLearnerTest
    {
        [TestMethod]
        public void RegressionStackingEnsembleLearner_Learn()
        {
            var learners = new IIndexedLearner<double>[]
            {
                new RegressionDecisionTreeLearner(2),
                new RegressionDecisionTreeLearner(5),
                new RegressionDecisionTreeLearner(7),
                new RegressionDecisionTreeLearner(9)
            };

            var sut = new RegressionStackingEnsembleLearner(learners, new RandomCrossValidation<double>(5, 23),
                new RegressionDecisionTreeLearner(9), false);

            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.06951934687172627, actual, 0.0001);
        }

        [TestMethod]
        public void RegressionStackingEnsembleLearner_Learn_Keep_Original_Features()
        {
            var learners = new IIndexedLearner<double>[]
            {
                new RegressionDecisionTreeLearner(2),
                new RegressionDecisionTreeLearner(5),
                new RegressionDecisionTreeLearner(7),
                new RegressionDecisionTreeLearner(9)
            };

            var sut = new RegressionStackingEnsembleLearner(learners, new RandomCrossValidation<double>(5, 23),
                new RegressionDecisionTreeLearner(9), true);

            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.066184865331534531, actual, 0.0001);
        }

        [TestMethod]
        public void RegressionStackingEnsembleLearner_Learn_Indexed()
        {
            var learners = new IIndexedLearner<double>[]
            {
                new RegressionDecisionTreeLearner(2),
                new RegressionDecisionTreeLearner(5),
                new RegressionDecisionTreeLearner(7),
                new RegressionDecisionTreeLearner(9)
            };

            var sut = new RegressionStackingEnsembleLearner(learners, new RandomCrossValidation<double>(5, 23),
                new RegressionDecisionTreeLearner(9), false);

            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var indices = Enumerable.Range(0, 25).ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.133930222950635, actual, 0.0001);
        }
    }
}
