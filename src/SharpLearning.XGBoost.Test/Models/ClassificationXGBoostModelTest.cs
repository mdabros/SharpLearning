using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using SharpLearning.XGBoost.Learners;
using SharpLearning.XGBoost.Models;
using SharpLearning.XGBoost.Test.Properties;

namespace SharpLearning.XGBoost.Test.Learners
{
    [TestClass]
    public class ClassificationXGBoostModelTest
    {
        readonly double m_delta = 0.0000001;

        [TestMethod]
        public void ClassificationXGBoostModel_Save_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var leaner = new ClassificationXGBoostLearner(estimaters: 2);
            var modelFilePath = "model.xgb";

            using (var sutPreSave = leaner.Learn(observations, targets))
            {
                AssertModel(observations, targets, sutPreSave);
                sutPreSave.Save(modelFilePath);
            }

            using (var sutAfterSave = ClassificationXGBoostModel.Load(modelFilePath))
            {
                AssertModel(observations, targets, sutAfterSave);
            }
        }

        void AssertModel(Containers.Matrices.F64Matrix observations, double[] targets, ClassificationXGBoostModel model)
        {
            var predictions = model.Predict(observations);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.17757009345794392, actual, m_delta);
        }
    }
}
