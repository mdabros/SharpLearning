﻿using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.GBMDecisionTree;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.GradientBoost.Test.GBMDecisionTree
{
    [TestClass]
    public class GBMDecisionTreeLearnerTest
    {
        [ExpectedException(typeof(ArgumentException))]
        [TestMethod]
        public void GBMDecisionTreeLearner_Constructor_MaximumTreeDepth()
        {
            new GBMDecisionTreeLearner(0);
        }

        [ExpectedException(typeof(ArgumentException))]
        [TestMethod]
        public void GBMDecisionTreeLearner_Constructor_MinimumSplitSize()
        {
            new GBMDecisionTreeLearner(1, 0);
        }

        [ExpectedException(typeof(ArgumentException))]
        [TestMethod]
        public void GBMDecisionTreeLearner_Constructor_MinimumInformationGain()
        {
            new GBMDecisionTreeLearner(1, 1, 0.0);
        }

        [ExpectedException(typeof(ArgumentException))]
        [TestMethod]
        public void GBMDecisionTreeLearner_Constructor_FeaturePrSplit()
        {
            new GBMDecisionTreeLearner(1, 1, 0.1, -1);
        }

        [ExpectedException(typeof(ArgumentNullException))]
        [TestMethod]
        public void GBMDecisionTreeLearner_Constructor_Loss_Null()
        {
            new GBMDecisionTreeLearner(1, 1, 1.0, 1, null, false);
        }
      
        [TestMethod]
        public void GBMDecisionTreeLearner_Learn()
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
            
            var predictions = tree.Predict(observations);
            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.0046122425037232661, actual);
        }
    }
}
