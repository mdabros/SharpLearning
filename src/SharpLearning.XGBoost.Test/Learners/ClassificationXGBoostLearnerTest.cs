using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using SharpLearning.XGBoost.Learners;
using SharpLearning.XGBoost.Test.Properties;

namespace SharpLearning.XGBoost.Test.Learners
{
    [TestClass]
    public class ClassificationXGBoostLearnerTest
    {
        readonly double m_delta = 0.0000001;

        [TestMethod]
        public void ClassificationXGBoostLearner_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationXGBoostLearner(estimaters: 2);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.17757009345794392, error, m_delta);
        }
    }
}
