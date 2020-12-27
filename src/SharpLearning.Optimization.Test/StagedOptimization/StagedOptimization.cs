namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public static class StagedOptimization
    {
        public static IParameterSpecsBuilder New()
        {
            return new AddParameterSpecBuilder();
        }
    }
}
