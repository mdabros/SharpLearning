namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public interface IOptimizerBuilder
    {
        IOptimizerBuilder AddOptimizer(CreateOptimizer createOptimizer);
        IOptimizer BuildOptimizer();
        IRunOptimizerBuilder BuildObjectiveFunction();
    }
}
