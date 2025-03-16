using SharpLearning.AdaBoost.Learners;
using SharpLearning.Common.Interfaces;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.RandomForest.Learners;
using System.Collections.Generic;

namespace SharpLearning.Benchmarks;

public static class DefaultLearners
{
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
}
