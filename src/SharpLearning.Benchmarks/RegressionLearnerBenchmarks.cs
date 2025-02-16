using SharpLearning.DecisionTrees.Learners;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.RandomForest.Learners;
using SharpLearning.GradientBoost.Learners;

namespace SharpLearning.Benchmarks;

public class RegressionDecisionTreeLearnerBenchmark()
    : RegressionLearnerBenchmark(() => new RegressionDecisionTreeLearner());

public class RegressionAdaboostLearnerBenchmark()
    : RegressionLearnerBenchmark(() => new RegressionAdaBoostLearner());

public class RegressionRandomForestLearnerBenchmark()
    : RegressionLearnerBenchmark(() => new RegressionRandomForestLearner());

public class RegressionExtremelyRandomizedLearnerBenchmark()
    : RegressionLearnerBenchmark(() => new RegressionExtremelyRandomizedTreesLearner());

public class RegressionSquareLossGradientBoostLearnerBenchmark()
    : RegressionLearnerBenchmark(() => new RegressionSquareLossGradientBoostLearner());
