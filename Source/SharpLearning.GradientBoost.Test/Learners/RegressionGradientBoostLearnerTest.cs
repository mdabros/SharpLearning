using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.Metrics.Regression;
using SharpLearning.Metrics.Classification;
using System.Diagnostics;
using System.Linq;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.GradientBoost.Loss;

namespace SharpLearning.GradientBoost.Test.Learners
{
    [TestClass]
    public class RegressionGradientBoostLearnerTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_Iterations()
        {
            new RegressionGradientBoostLearner(0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_LearningRate()
        {
            new RegressionGradientBoostLearner(1, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_MaximumTreeDepth()
        {
            new RegressionGradientBoostLearner(1, 1, -1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_MinimumSplitSize()
        {
            new RegressionGradientBoostLearner(1, 1, 1, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_MinimumInformationGain()
        {
            new RegressionGradientBoostLearner(1, 1, 1, 1, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_SubSampleRatio_TooLow()
        {
            new RegressionGradientBoostLearner(1, 1, 1, 1, 1, 0.0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_SubSampleRatio_TooHigh()
        {
            new RegressionGradientBoostLearner(1, 1, 1, 1, 1, 1.1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RegressionGradientBoostLearner_Constructor_Loss()
        {
            new RegressionGradientBoostLearner(1, 1, 1, 1, 1, 1.0, null, 1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RegressionGradientBoostLearner_Constructor_SubSampleRatio_NumberOfThreads()
        {
            new RegressionGradientBoostLearner(1, 1, 1, 1, 1, 1.0, new GradientBoostSquaredLoss(), 0);
        }
    }
}
