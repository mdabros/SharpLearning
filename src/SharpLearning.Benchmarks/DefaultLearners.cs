using System.Collections.Generic;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.AdaBoost.Models;
using SharpLearning.Common.Interfaces;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.GradientBoost.Models;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Models;

namespace SharpLearning.Benchmarks;

public static class DefaultLearners
{
    const string NameSeparator = "_";

    // Define classification learners here. Use default parameters for benchmarks.
    public static readonly IReadOnlyDictionary<string, ILearner<double>> LearnerNameToLearnerClassification =
        new Dictionary<string, ILearner<double>>()
    {
        { nameof(ClassificationDecisionTreeLearner), new ClassificationDecisionTreeLearner() },
        { nameof(ClassificationAdaBoostLearner), new ClassificationAdaBoostLearner() },
        { nameof(ClassificationRandomForestLearner), new ClassificationRandomForestLearner() },
        { nameof(ClassificationExtremelyRandomizedTreesLearner), new ClassificationExtremelyRandomizedTreesLearner() },
        { nameof(ClassificationBinomialGradientBoostLearner), new ClassificationBinomialGradientBoostLearner() },
    };

    // Define regression learners here. Use default parameters for benchmarks.
    public static readonly IReadOnlyDictionary<string, ILearner<double>> LearnerNameToLearnerRegression =
        new Dictionary<string, ILearner<double>>()
    {
        { nameof(RegressionDecisionTreeLearner), new RegressionDecisionTreeLearner() },
        { nameof(RegressionAdaBoostLearner), new RegressionAdaBoostLearner() },
        { nameof(RegressionRandomForestLearner), new RegressionRandomForestLearner() },
        { nameof(RegressionExtremelyRandomizedTreesLearner), new RegressionExtremelyRandomizedTreesLearner() },
        { nameof(RegressionAbsoluteLossGradientBoostLearner), new RegressionAbsoluteLossGradientBoostLearner() },
        { nameof(RegressionHuberLossGradientBoostLearner), new RegressionHuberLossGradientBoostLearner() },
        { nameof(RegressionQuantileLossGradientBoostLearner), new RegressionQuantileLossGradientBoostLearner() },
        { nameof(RegressionSquareLossGradientBoostLearner), new RegressionSquareLossGradientBoostLearner() },
    };

    // Map learner names to model names for classification. Some learners return the same model type,
    // so suffixing with the learner name.
    public static readonly IReadOnlyDictionary<string, string> LearnerNameToModelNameClassification =
        new Dictionary<string, string>()
    {
        { nameof(ClassificationDecisionTreeLearner), nameof(ClassificationDecisionTreeModel) },
        { nameof(ClassificationAdaBoostLearner), nameof(ClassificationAdaBoostModel) },
        { nameof(ClassificationRandomForestLearner), nameof(ClassificationForestModel) + NameSeparator + nameof(ClassificationRandomForestLearner) },
        { nameof(ClassificationExtremelyRandomizedTreesLearner), nameof(ClassificationForestModel) + NameSeparator + nameof(ClassificationExtremelyRandomizedTreesLearner)},
        { nameof(ClassificationBinomialGradientBoostLearner), nameof(ClassificationGradientBoostModel) },
    };

    // Map learner names to model names for regression. Some learners return the same model type,
    // so suffixing with the learner name.
    public static readonly IReadOnlyDictionary<string, string> LearnerNameToModelNameRegression =
        new Dictionary<string, string>()
    {
        { nameof(RegressionDecisionTreeLearner), nameof(RegressionDecisionTreeModel) },
        { nameof(RegressionAdaBoostLearner), nameof(RegressionAdaBoostModel) },
        { nameof(RegressionRandomForestLearner), nameof(RegressionForestModel) + NameSeparator + nameof(RegressionRandomForestLearner)},
        { nameof(RegressionExtremelyRandomizedTreesLearner), nameof(RegressionForestModel) + NameSeparator + nameof(RegressionExtremelyRandomizedTreesLearner) },
        { nameof(RegressionAbsoluteLossGradientBoostLearner), nameof(RegressionGradientBoostModel) + NameSeparator + nameof(RegressionAbsoluteLossGradientBoostLearner) },
        { nameof(RegressionHuberLossGradientBoostLearner), nameof(RegressionGradientBoostModel) + NameSeparator + nameof(RegressionHuberLossGradientBoostLearner) },
        { nameof(RegressionQuantileLossGradientBoostLearner), nameof(RegressionGradientBoostModel) + NameSeparator + nameof(RegressionQuantileLossGradientBoostLearner) },
        { nameof(RegressionSquareLossGradientBoostLearner), nameof(RegressionGradientBoostModel) + NameSeparator + nameof(RegressionSquareLossGradientBoostLearner) },
    };
}
