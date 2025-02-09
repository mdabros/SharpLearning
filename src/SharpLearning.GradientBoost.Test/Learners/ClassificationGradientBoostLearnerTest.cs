using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.GradientBoost.Test.Learners;

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
    [ExpectedException(typeof(ArgumentException))]
    public void ClassificationGradientBoostLearner_Constructor_FeaturePrSplit()
    {
        new ClassificationGradientBoostLearner(1, 1, 1, 1, 1, 1.0, -1);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void ClassificationGradientBoostLearner_Constructor_Loss()
    {
        new ClassificationGradientBoostLearner(1, 1, 1, 1, 1, 1.0, 1, null, false);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void ClassificationGradientBoostLearner_LearnWithEarlyStopping_ToFewIterations()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var splitter = new StratifiedTrainingTestIndexSplitter<double>(0.6, 1234);
        var split = splitter.SplitSet(observations, targets);

        var sut = new ClassificationBinomialGradientBoostLearner(10, 0.01, 9, 1, 1e-6, .5, 1);
        var evaluator = new TotalErrorClassificationMetric<double>();

        var model = sut.LearnWithEarlyStopping(split.TrainingSet.Observations, split.TrainingSet.Targets,
            split.TestSet.Observations, split.TestSet.Targets, evaluator, 10);
    }

    [TestMethod]
    public void ClassificationGradientBoostLearner_LearnWithEarlyStopping()
    {
        var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

        var splitter = new StratifiedTrainingTestIndexSplitter<double>(0.6, 1234);
        var split = splitter.SplitSet(observations, targets);

        var sut = new ClassificationBinomialGradientBoostLearner(100, 0.01, 9, 1, 1e-6, .5, 0, false);
        var evaluator = new TotalErrorClassificationMetric<double>();

        var model = sut.LearnWithEarlyStopping(split.TrainingSet.Observations, split.TrainingSet.Targets,
            split.TestSet.Observations, split.TestSet.Targets, evaluator, 10);

        var predictions = model.Predict(split.TestSet.Observations);
        var actual = evaluator.Error(split.TestSet.Targets, predictions);

        Assert.AreEqual(0.16279069767441862, actual, 0.000001);
        Assert.AreEqual(90, model.Trees.First().ToArray().Length);
    }
}
