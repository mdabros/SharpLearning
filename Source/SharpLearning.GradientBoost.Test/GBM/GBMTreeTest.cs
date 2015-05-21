using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.InputOutput.Csv;
using System.Linq;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.GBM;

namespace SharpLearning.GradientBoost.Test.GBM
{
    [TestClass]
    public class GBMTreeTest
    {
        [TestMethod]
        public void GBMTree_AddRawFeatureImportances()
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

            var sut = new GBMDecisionTreeLearner(10);
            var tree = sut.Learn(observations, targets, targets, orderedElements, inSample, targets.Length);

            var actual = new double[observations.GetNumberOfColumns()];
            tree.AddRawVariableImportances(actual);

            var expected = new double[] { 0.0, 105017.48701572006 };
            Assert.AreEqual(expected.Length, actual.Length);
            Assert.AreEqual(expected[0], actual[0], 0.01);
            Assert.AreEqual(expected[1], actual[1], 0.01);
        }
    }
}
