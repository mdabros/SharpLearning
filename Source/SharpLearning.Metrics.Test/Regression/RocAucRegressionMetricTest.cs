using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Regression;
using System.Linq;

namespace SharpLearning.Metrics.Test.Regression
{
    /// <summary>
    /// Summary description for RocAucRegressionMetricTest
    /// </summary>
    [TestClass]
    public class RocAucRegressionMetricTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RocAucRegressionMetric_Error_Not_Binary()
        {
            var targets = new double[] { 0, 1, 2 };
            var probabilities = new double[0];

            var sut = new RocAucRegressionMetric(1);
            var actual = sut.Error(targets, probabilities);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RocAucRegressionMetric_Error_TargetToBinaryMapping_Has_More_Than_Two_Values()
        {
            new RocAucRegressionMetric(1, new Dictionary<double, double> { { 0.0, 0.0 }, { 1.0, 1.0 }, { 2.0, 2.0 }, { 3.0, 3.0 } });
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RocAucRegressionMetric_Error_TargetToBinaryMapping_Has_Less_Than_Two_Values()
        {
            new RocAucRegressionMetric(1, new Dictionary<double, double> { { 0.0, 0.0 }, { 1.0, 0.0 } });
        }

        [TestMethod]
        public void RocAucRegressionMetric_Error_Using_Mapping()
        {
            var targets = new double[] { 1, 0, 1, 0, 0, 0, 1, 3, 0, 0, 1, 1, 3, 0, 1, 1, 3, 2, 0, 0, 2, 1 };
            var probabilities = new double[] { 0.052380952, 0.020725389, 0.993377483, 0.020725389, 0.020725389, 0.111111111, 0.193377483, 0.793377483, 0.020725389, 0.012345679, 0.885860173, 0.714285714, 0.985860173, 0.020725389, 0.985860173, 0.993377483, 0.993377483, 0.954545455, 0.020725389, 0.020725389, 0.985860173, 0.985860173 };

            var sut = new RocAucRegressionMetric(1, new Dictionary<double, double> { { 0.0, 0.0 }, { 1.0, 1.0 }, { 2.0, 1.0 }, { 3.0, 1.0 } });
            var actual = sut.Error(targets, probabilities);
            Assert.AreEqual(0.0085470085470086277, actual, 0.00001);
        }

        [TestMethod]
        public void RocAucRegressionMetric_Error_No_Error()
        {
            var targets = new double[] { 0, 1 };
            var probabilities = new double[] { 0.0, 1 };
            var sut = new RocAucRegressionMetric(1);
            var actual = sut.Error(targets, probabilities);

            Assert.AreEqual(0.0, actual);
        }

        [TestMethod]
        public void RocAucRegressionMetric_Error()
        {
            var targets = new double[] { 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1 };
            var probabilities = new double[] { 0.052380952 , 0.020725389, 0.993377483, 0.020725389, 0.020725389, 0.111111111, 0.193377483, 0.793377483, 0.020725389, 0.012345679, 0.885860173, 0.714285714, 0.985860173, 0.020725389, 0.985860173, 0.993377483, 0.993377483, 0.954545455, 0.020725389, 0.020725389, 0.985860173, 0.985860173 };

            var sut = new RocAucRegressionMetric(1);
            var actual = sut.Error(targets, probabilities);
            Assert.AreEqual(0.0085470085470086277, actual, 0.00001);
        }

        [TestMethod]
        public void RocAucRegressionMetric_Error_Random()
        {
            var positives = Enumerable.Range(0, 800).Select(s => 1.0).ToList();
            var negatives = Enumerable.Range(0, 800).Select(s => 0.0).ToList();
            positives.AddRange(negatives);
            var targets = positives.ToArray();

            var random = new Random(42);
            var probabilities = targets.Select(s => random.NextDouble()).ToArray();


            var sut = new RocAucRegressionMetric(1);
            var actual = sut.Error(targets, probabilities);

            Assert.AreEqual(0.488959375, actual, 0.0001);
        }

        [TestMethod]
        public void RocAucRegressionMetric_Error_Always_Negative()
        {
            var positives = Enumerable.Range(0, 200).Select(s => 1.0).ToList();
            var negatives = Enumerable.Range(0, 800).Select(s => 0.0).ToList();
            positives.AddRange(negatives);
            var targets = positives.ToArray();

            var probabilities = targets
                .Select(s => 0.0)
                .ToArray();


            var sut = new RocAucRegressionMetric(1);
            var actual = sut.Error(targets, probabilities);

            Assert.AreEqual(0.5, actual, 0.0001);
        }

        [TestMethod]
        public void RocAucRegressionMetric_Error_Always_Positve()
        {
            var positives = Enumerable.Range(0, 800).Select(s => 1.0).ToList();
            var negatives = Enumerable.Range(0, 200).Select(s => 0.0).ToList();
            positives.AddRange(negatives);
            var targets = positives.ToArray();

            var probabilities = targets
                .Select(s => 1.0)
                .ToArray();

            var sut = new RocAucRegressionMetric(1);
            var actual = sut.Error(targets, probabilities);

            Assert.AreEqual(0.5, actual, 0.0001);
        }
    }
}
