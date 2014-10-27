using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Arithmetic;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Linear.Optimization;
using SharpLearning.Linear.Test.Properties;
using SharpLearning.Metrics.Regression;
using System.IO;
using System.Linq;

namespace SharpLearning.Linear.Test.Optimization
{
    [TestClass]
    public class StochasticGradientDescentTest
    {
        [TestMethod]
        public void StochasticGradientDescent_Optimize_Linear_Regression()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Housing));
            var observations = parser.EnumerateRows("Size", "Rooms").ToF64Matrix();
            var targets = parser.EnumerateRows("Price").ToF64Vector();
            
            var sut = new StochasticGradientDescent();
            var theta = sut.Optimize(observations, targets);

            var bias = Enumerable.Range(0, targets.Length)
                .Select(b => 1.0).ToArray();
            var x = bias.CombineCols(observations);

            var metric = new MeanSquaredErrorRegressionMetric();
            var predictions = observations.Multiply(theta);
            var error = metric.Error(targets, predictions);

            Assert.AreEqual(207378680059.91843, error, 0.001);
            Assert.AreEqual(335724.3400124929, theta[0], 0.001);
            Assert.AreEqual(100826.3400268207, theta[1], 0.001);
            Assert.AreEqual(4206.8294406613268, theta[2], 0.001);
        }
    }
}
