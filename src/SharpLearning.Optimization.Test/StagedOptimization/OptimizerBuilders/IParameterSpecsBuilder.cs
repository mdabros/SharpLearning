namespace SharpLearning.Optimization.Test.StagedOptimization.OptimizerBuilders
{
    public interface IParameterSpecsBuilder
    {
        IParameterSpecsBuilder AddParameterSpec(IParameterSpec parameterSpec);
        IParameterSpec[] BuildParameterSpecs();
        IOptimizerBuilder BuildOptimizer();
    }
}
