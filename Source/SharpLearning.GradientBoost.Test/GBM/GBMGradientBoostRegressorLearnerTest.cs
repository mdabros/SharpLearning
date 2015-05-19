using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.GradientBoost.GBM;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.GradientBoost.Test.GBM
{
    [TestClass]
    public class GBMGradientBoostRegressorLearnerTest
    {
        [TestMethod]
        public void GBMGradientBoostRegressorLearner_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();

            var sut = new GBMGradientBoostRegressorLearner(50, 0.1, 3, 1, 1e-6, 1.0, new GBMSquaredLoss(), 1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);
            
            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.023740932093339349, actual);
        }

        [TestMethod]
        public void GBMGradientBoostRegressorLearner_Stochastic_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();

            var sut = new GBMGradientBoostRegressorLearner(50, 0.1, 3, 1, 1e-6, .5, new GBMSquaredLoss(), 1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038289922749043533, actual);
        }
    }
}
