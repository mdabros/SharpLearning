namespace SharpLearning.Optimization.Test.StagedOptimization.OptimizerBuilders
{
    public interface IOptimizerBuilder
    {
        IOptimizerBuilder AddOptimizer(CreateOptimizer createOptimizer);
        IOptimizer BuildOptimizer();
        IRunOptimizerBuilder BuildObjectiveFunction();
    }
}
