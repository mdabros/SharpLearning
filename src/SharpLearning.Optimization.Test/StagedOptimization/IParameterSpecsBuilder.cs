namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public interface IParameterSpecsBuilder
    {
        IParameterSpecsBuilder AddParameterSpec(IParameterSpec parameterSpec);
        IParameterSpec[] BuildParameterSpecs();
        IOptimizerBuilder BuildOptimizer();
    }
}
