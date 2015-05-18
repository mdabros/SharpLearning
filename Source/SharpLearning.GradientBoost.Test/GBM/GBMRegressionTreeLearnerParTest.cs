using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.GBM;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using System.IO;
using System.Linq;

namespace SharpLearning.GradientBoost.Test.GBM
{
    [TestClass]
    public class GBMRegressionTreeLearnerParTest
    {
        [TestMethod]
        public void GBMRegressionTreeLearnerPar_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();

            var inSample = targets.Select(t => true).ToArray();
            var orderedElements = new int[observations.GetNumberOfColumns()][];
            var rows = observations.GetNumberOfRows();

            for (int i = 0; i < observations.GetNumberOfColumns(); i++)
			{
			    var feature = observations.GetColumn(i);
                var indices = Enumerable.Range(0, rows).ToArray();
                feature.SortWith(indices);
                orderedElements[i] = indices;
			}

            var sum = targets.Sum();
            var sumSquare = targets.Select(t => t * t).Sum();

            var sut = new GBMRegressionTreeLearnerPar(10);

            var tree = sut.Learn(observations, targets, orderedElements, inSample, sum, sumSquare, targets.Length);
            tree.TraceNodesDepth();
            var predictions = tree.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.0046122425037232661, actual);
        }
    }
}
