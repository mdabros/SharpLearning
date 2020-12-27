namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public interface IRunOptimizerBuilder
    {
        IRunOptimizerBuilder AddObjective(ObjectiveFunction objectiveFunction);
        RunOptimizer BuildRunOptimizer();
    }
}
