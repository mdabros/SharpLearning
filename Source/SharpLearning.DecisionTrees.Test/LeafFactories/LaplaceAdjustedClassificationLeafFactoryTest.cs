using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.LeafFactories;
using SharpLearning.DecisionTrees.Nodes;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.DecisionTrees.Test.LeafFactories
{
    [TestClass]
    public class LaplaceAdjustedClassificationLeafFactoryTest
    {
        [TestMethod]
        public void LaplaceAdjustedClassificationLeafValueFactory_Create()
        {
            var values = new double[] { 1, 1, 1, 2, 2 };
            var sut = new LaplaceAdjustedClassificationLeafFactory();
            var actual = sut.Create(new ContinousBinaryDecisionNode(), values, values.Distinct().ToArray());
            var expectedProbabilities = new ProbabilityPrediction(1, new Dictionary<double, double> { { 1.0, 0.571428571428571}, { 2.0, 0.428571428571429 } });
            var actualProbabilities = actual.PredictProbability(values);

            Assert.AreEqual(expectedProbabilities, actualProbabilities);
            Assert.AreEqual(1, actual.Value);
            Assert.AreEqual(-1, actual.FeatureIndex);
            Assert.IsTrue(actual.IsLeaf());
            Assert.IsNotNull(actual.Parent);
        }

        [TestMethod]
        public void LaplaceAdjustedClassificationLeafValueFactory_Create_Equal()
        {
            var values = new double[] { 1, 1, 2, 2 };
            var sut = new LaplaceAdjustedClassificationLeafFactory();
            var actual = sut.Create(new ContinousBinaryDecisionNode(), values, values.Distinct().ToArray());
            var expectedProbabilities = new ProbabilityPrediction(1, new Dictionary<double, double> { { 1.0, 0.5 }, { 2.0, 0.5 } });
            var actualProbabilities = actual.PredictProbability(values);

            Assert.AreEqual(expectedProbabilities, actualProbabilities);
            Assert.AreEqual(1, actual.Value);
            Assert.AreEqual(-1, actual.FeatureIndex);
            Assert.IsTrue(actual.IsLeaf());
            Assert.IsNotNull(actual.Parent);
        }

        [TestMethod]
        public void LaplaceAdjustedClassificationLeafValueFactory_Create_Interval()
        {
            var values = new double[] { 1, 1, 1, 2, 2 };
            var sut = new LaplaceAdjustedClassificationLeafFactory();
            var actual = sut.Create(new ContinousBinaryDecisionNode(), values, values.Distinct().ToArray(), Interval1D.Create(2, 5));
            var expectedProbabilities = new ProbabilityPrediction(2, new Dictionary<double, double> { { 1.0, 0.4 }, { 2.0, 0.6 } });
            var actualProbabilities = actual.PredictProbability(values);

            Assert.AreEqual(expectedProbabilities, actualProbabilities);
            Assert.AreEqual(2, actual.Value);
            Assert.AreEqual(-1, actual.FeatureIndex);
            Assert.IsTrue(actual.IsLeaf());
            Assert.IsNotNull(actual.Parent);
        }
    }
}
