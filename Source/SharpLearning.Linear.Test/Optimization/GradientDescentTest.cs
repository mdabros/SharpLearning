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
    public class GradientDescentTest
    {
        [TestMethod]
        public void GradientDescent_Optimize_Linear_Regression()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Housing));
            var observations = parser.EnumerateRows("Size", "Rooms").ToF64Matrix();
            var targets = parser.EnumerateRows("Price").ToF64Vector();
            
            var sut = new GradientDescent(0.01, 400);
            var theta = sut.Optimize(observations, targets);

            var bias = Enumerable.Range(0, targets.Length)
                .Select(b => 1.0).ToArray();
            var x = bias.CombineCols(observations);

            var metric = new MeanSquaredErrorRegressionMetric();
            var predictions = observations.Multiply(theta);
            var error = metric.Error(targets, predictions);

            Assert.AreEqual(206250227200.725, error, 0.001);
            Assert.AreEqual(334302.063989956, theta[0], 0.001);
            Assert.AreEqual(100087.116006087, theta[1], 0.001);
            Assert.AreEqual(3673.54844680899, theta[2], 0.001);
        }
    }
}
