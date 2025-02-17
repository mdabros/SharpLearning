using SharpLearning.DecisionTrees.Learners;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.RandomForest.Learners;
using SharpLearning.GradientBoost.Learners;

namespace SharpLearning.Benchmarks;

public static partial class Benchmarks
{
    public static partial class Regression
    {
        public class DecisionTreeLearner()
            : RegressionLearnerBenchmark(() => new RegressionDecisionTreeLearner());

        public class AdaboostLearner()
            : RegressionLearnerBenchmark(() => new RegressionAdaBoostLearner());

        public class RandomForestLearner()
            : RegressionLearnerBenchmark(() => new RegressionRandomForestLearner());

        public class ExtremelyRandomizedTreeLearner()
            : RegressionLearnerBenchmark(() => new RegressionExtremelyRandomizedTreesLearner());

        public class SquareLossGradientBoostLearner()
            : RegressionLearnerBenchmark(() => new RegressionSquareLossGradientBoostLearner());
    }
}
