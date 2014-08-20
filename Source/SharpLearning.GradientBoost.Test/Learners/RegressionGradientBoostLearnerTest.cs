using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using System.IO;

namespace SharpLearning.GradientBoost.Test.Learners
{
    [TestClass]
    public class RegressionGradientBoostLearnerTest
    {
        [TestMethod]
        public void RegressionGradientBoostLearner_Learn_AptitudeData()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new RegressionGradientBoostLearner();

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.052593937296318408, actual);
        }

        [TestMethod]
        public void RegressionGradientBoostLearner_Learn_Glass()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionGradientBoostLearner();

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.18457759227201861, actual);
        }

        //[TestMethod]
        //public void RegressionGradientBoostLearner_Learn_Glass_Indexed()
        //{
        //    var parser = new CsvParser(() => new StringReader(Resources.Glass));
        //    var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
        //    var targets = parser.EnumerateRows("Target").ToF64Vector();

        //    var sut = new RegressionGradientBoostLearner(10, 1, 5);

        //    var indices = Enumerable.Range(0, targets.Length).ToArray();
        //    indices.Shuffle(new Random(42));
        //    indices = indices.Take((int)(targets.Length * 0.7))
        //        .ToArray();

        //    var model = sut.Learn(observations, targets, indices);
        //    var predictions = model.Predict(observations);
        //    var indexedPredictions = predictions.GetIndices(indices);
        //    var indexedTargets = targets.GetIndices(indices);

        //    var evaluator = new MeanAbsolutErrorRegressionMetric();
        //    var actual = evaluator.Error(indexedTargets, indexedPredictions);

        //    Assert.AreEqual(0.22181054803405248, actual);
        //}

    }
}
