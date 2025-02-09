using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Common.Interfaces;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Ensemble.Test.Learners;

[TestClass]
public class RegressionStackingEnsembleLearnerTest
{
    [TestMethod]
    public void RegressionStackingEnsembleLearner_Learn()
    {
        var learners = new IIndexedLearner<double>[]
        {
            new RegressionDecisionTreeLearner(2),
            new RegressionDecisionTreeLearner(5),
            new RegressionDecisionTreeLearner(7),
            new RegressionDecisionTreeLearner(9)
        };

        var sut = new RegressionStackingEnsembleLearner(learners,
            new RegressionDecisionTreeLearner(9),
            new RandomCrossValidation<double>(5, 23), false);

        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var model = sut.Learn(observations, targets);
        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.06951934687172627, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionStackingEnsembleLearner_CreateMetaFeatures_Then_Learn()
    {
        var learners = new IIndexedLearner<double>[]
        {
            new RegressionDecisionTreeLearner(2),
            new RegressionDecisionTreeLearner(5),
            new RegressionDecisionTreeLearner(7),
            new RegressionDecisionTreeLearner(9)
        };

        var sut = new RegressionStackingEnsembleLearner(learners,
            new RegressionDecisionTreeLearner(9),
            new RandomCrossValidation<double>(5, 23), false);

        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var metaObservations = sut.LearnMetaFeatures(observations, targets);
        var model = sut.LearnStackingModel(observations, metaObservations, targets);

        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.06951934687172627, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionStackingEnsembleLearner_Learn_Keep_Original_Features()
    {
        var learners = new IIndexedLearner<double>[]
        {
            new RegressionDecisionTreeLearner(2),
            new RegressionDecisionTreeLearner(5),
            new RegressionDecisionTreeLearner(7),
            new RegressionDecisionTreeLearner(9)
        };

        var sut = new RegressionStackingEnsembleLearner(learners,
            new RegressionDecisionTreeLearner(9),
            new RandomCrossValidation<double>(5, 23), true);

        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var model = sut.Learn(observations, targets);
        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.066184865331534531, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionStackingEnsembleLearner_Learn_Indexed()
    {
        var learners = new IIndexedLearner<double>[]
        {
            new RegressionDecisionTreeLearner(2),
            new RegressionDecisionTreeLearner(5),
            new RegressionDecisionTreeLearner(7),
            new RegressionDecisionTreeLearner(9)
        };

        var sut = new RegressionStackingEnsembleLearner(learners,
            new RegressionDecisionTreeLearner(9),
            new RandomCrossValidation<double>(5, 23), false);

        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var indices = Enumerable.Range(0, 25).ToArray();

        var model = sut.Learn(observations, targets, indices);
        var predictions = model.Predict(observations);

        var evaluator = new MeanSquaredErrorRegressionMetric();
        var actual = evaluator.Error(targets, predictions);

        Assert.AreEqual(0.133930222950635, actual, 0.0001);
    }
}
