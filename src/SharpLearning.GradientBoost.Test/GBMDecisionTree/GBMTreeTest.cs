using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.GBMDecisionTree;

namespace SharpLearning.GradientBoost.Test.GBMDecisionTree
{
    [TestClass]
    public class GBMTreeTest
    {
        [TestMethod]
        public void GBMTree_AddRawFeatureImportances()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var inSample = targets.Select(t => true).ToArray();
            var orderedElements = new int[observations.ColumnCount][];
            var rows = observations.RowCount;

            for (int i = 0; i < observations.ColumnCount; i++)
            {
                var feature = observations.Column(i);
                var indices = Enumerable.Range(0, rows).ToArray();
                feature.SortWith(indices);
                orderedElements[i] = indices;
            }

            var sut = new GBMDecisionTreeLearner(10);
            var tree = sut.Learn(observations, targets, targets, targets, orderedElements, inSample);

            var actual = new double[observations.ColumnCount];
            tree.AddRawVariableImportances(actual);

            var expected = new double[] { 0.0, 105017.48701572006 };
            Assert.AreEqual(expected.Length, actual.Length);
            Assert.AreEqual(expected[0], actual[0], 0.01);
            Assert.AreEqual(expected[1], actual[1], 0.01);
        }
    }
}
