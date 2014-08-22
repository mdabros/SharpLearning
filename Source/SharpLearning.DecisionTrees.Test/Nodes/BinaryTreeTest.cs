using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.DecisionTrees.Test.Properties;
using SharpLearning.DecisionTrees.Learners;
using System.Linq;

namespace SharpLearning.DecisionTrees.Test.Nodes
{
    [TestClass]
    public class BinaryTreeTest
    {
        [TestMethod]
        public void BinaryTree_LeafRegionIndices()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2")
                .Take(10).ToF64Matrix();
            var targets = parser.EnumerateRows("T")
                .Take(10).ToF64Vector();

            var rows = targets.Length;

            var learner = new RegressionDecisionTreeLearner(1);
            var sut = learner.Learn(observations, targets).Tree;

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            var actual = sut.LeafRegionIndices(observations, indices);
            
            var expected = new List<int>[]
            {
                new List<int> {1, 4, 7, 9},
                new List<int> {0, 2, 3, 5, 6, 8},
            };

            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                CollectionAssert.AreEqual(expected[i], actual[i]);
            }
        }
    }
}
