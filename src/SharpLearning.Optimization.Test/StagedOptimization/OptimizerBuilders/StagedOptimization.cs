namespace SharpLearning.Optimization.Test.StagedOptimization.OptimizerBuilders
{
    public static class CreateOptimizerBuilder
    {
        public static IParameterSpecsBuilder New()
        {
            return new AddParameterSpecBuilder();
        }
    }
}
