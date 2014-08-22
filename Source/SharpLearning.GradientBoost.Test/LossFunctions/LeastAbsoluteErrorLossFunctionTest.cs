using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.GradientBoost.LossFunctions;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.InputOutput.Csv;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.GradientBoost.Test.LossFunctions
{
    [TestClass]
    public class LeastAbsoluteErrorLossFunctionTest
    {
        [TestMethod]
        public void LeastAbsoluteErrorLossFunction_InitializeLoss()
        {
            var targets = new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9};
            var actual = new double[targets.Length];
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            var sut = new LeastAbsoluteErrorLossFunction(0.1);

            sut.InitializeLoss(targets, actual, indices);

            var expected = new double[] { 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void LeastAbsoluteErrorLossFunction_NegativeGradient()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var predictions = new double[] { 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0 };
            var actual = new double[targets.Length];
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            var sut = new LeastAbsoluteErrorLossFunction(0.1);

            sut.NegativeGradient(targets, predictions, actual, indices);

            var expected = new double[] { -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void LeastAbsoluteErrorLossFunction_IntialLoss_LearningRate()
        {
            var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var actual = new double[targets.Length];
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            var sut = new LeastAbsoluteErrorLossFunction(0.1);

            sut.InitializeLoss(targets, actual, indices);

            Assert.AreEqual(5.0, sut.InitialLoss);
            Assert.AreEqual(0.1, sut.LearningRate);
        }

        [TestMethod]
        public void LeastAbsoluteErrorLossFunction_UpdateModel()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            
            var observations = parser.EnumerateRows(v => v != "T")
                .Take(10).ToF64Matrix();
            var targets = parser.EnumerateRows("T")
                .Take(10).ToF64Vector();

            var actual = new double[targets.Length];
            var residuals = new double[targets.Length];
            var indices = Enumerable.Range(0, targets.Length).ToArray();

            var sut = new LeastAbsoluteErrorLossFunction(0.1);
            sut.InitializeLoss(targets, actual, indices);
            sut.NegativeGradient(targets, actual, residuals, indices);

            var model = new RegressionDecisionTreeLearner().Learn(observations, residuals);
            sut.UpdateModel(model.Tree, observations, targets, actual, indices);

            var expected = new double[] { 1.78401305, 1.6547943, 1.8630581, 1.8630581, 1.6547943, 
                1.76197595, 1.8630581, 1.6547943, 1.8630581, 1.6547943};
            
            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.00001);
            }
        }
    }
}
