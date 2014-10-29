using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Linear.Optimization;
using SharpLearning.Linear.Test.Properties;
using System.IO;

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
            
            var sut = new StochasticGradientDescent(0.001, 5000, 42, 1);
            var theta = sut.Optimize(observations, targets);

            Assert.AreEqual(336739.710490569, theta[0], 0.001);
            Assert.AreEqual(105731.26301175922, theta[1], 0.001);
            Assert.AreEqual(-4544.3634625597488, theta[2], 0.001);
        }
    }
}
