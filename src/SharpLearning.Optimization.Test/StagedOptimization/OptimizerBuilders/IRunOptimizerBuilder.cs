namespace SharpLearning.Optimization.Test.StagedOptimization.OptimizerBuilders
{
    public interface IRunOptimizerBuilder
    {
        IRunOptimizerBuilder AddObjective(ObjectiveFunction objectiveFunction);
        RunOptimizer BuildRunOptimizer();
    }
}
