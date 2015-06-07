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
    public class ClassificationGradientBoostLearnerTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ClassificationGradientBoostLearner_Constructor_Iterations()
        {
            new ClassificationGradientBoostLearner(0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ClassificationGradientBoostLearner_Constructor_LearningRate()
        {
            new ClassificationGradientBoostLearner(1, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ClassificationGradientBoostLearner_Constructor_MaximumTreeDepth()
        {
            new ClassificationGradientBoostLearner(1, 1, -1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ClassificationGradientBoostLearner_Constructor_MinimumSplitSize()
        {
            new ClassificationGradientBoostLearner(1, 1, 1, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ClassificationGradientBoostLearner_Constructor_MinimumInformationGain()
        {
            new ClassificationGradientBoostLearner(1, 1, 1, 1, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ClassificationGradientBoostLearner_Constructor_SubSampleRatio_TooLow()
        {
            new ClassificationGradientBoostLearner(1, 1, 1, 1, 1, 0.0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ClassificationGradientBoostLearner_Constructor_SubSampleRatio_TooHigh()
        {
            new ClassificationGradientBoostLearner(1, 1, 1, 1, 1, 1.1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void ClassificationGradientBoostLearner_Constructor_Loss()
        {
            new ClassificationGradientBoostLearner(1, 1, 1, 1, 1, 1.0, null, 1);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void ClassificationGradientBoostLearner_Constructor_SubSampleRatio_NumberOfThreads()
        {
            new ClassificationGradientBoostLearner(1, 1, 1, 1, 1, 1.0, new GradientBoostSquaredLoss(), 0);
        }
    }
}
