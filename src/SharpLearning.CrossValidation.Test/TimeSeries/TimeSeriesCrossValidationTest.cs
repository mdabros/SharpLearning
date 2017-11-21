using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.Test.Properties;
using SharpLearning.CrossValidation.TimeSeries;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.CrossValidation.Test.TimeSeries
{
    [TestClass]
    public class TimeSeriesCrossValidationTest
    {
        [TestMethod]
        public void TimeSeriesCrossValidation_Validate()
        {
            var targetName = "T";
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows(v => !v.Contains(targetName)).ToF64Matrix();
            var targets = parser.EnumerateRows(targetName).ToF64Vector();
            var initialTrainingSize = 5;

            var sut = new TimeSeriesCrossValidation<double>(initialTrainingSize);
            var timeSeriesPredictions = sut.Validate(new RegressionDecisionTreeLearner(), observations, targets);
            var timeSeriesTargets = targets.Skip(initialTrainingSize).ToArray();

            var metric = new MeanSquaredErrorRegressionMetric();
            var error = metric.Error(timeSeriesTargets, timeSeriesPredictions);

            Assert.AreEqual(0.097833163747046217, error, 0.00001);
        }
    }
}
