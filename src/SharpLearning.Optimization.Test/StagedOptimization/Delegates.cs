namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public delegate IOptimizer CreateOptimizer(IParameterSpec[] parameterSpecs);
    public delegate OptimizerResult ObjectiveFunction(double[] parameters);
    public delegate OptimizerResult[] RunOptimizer();
}
