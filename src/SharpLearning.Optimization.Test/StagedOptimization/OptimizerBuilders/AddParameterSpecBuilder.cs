using System.Collections.Generic;

namespace SharpLearning.Optimization.Test.StagedOptimization.OptimizerBuilders
{
    public class AddParameterSpecBuilder : IParameterSpecsBuilder
    {
        readonly List<IParameterSpec> m_parameterSpecs = new List<IParameterSpec>();

        public IParameterSpecsBuilder AddParameterSpec(IParameterSpec parameterSpec)
        {
            m_parameterSpecs.Add(parameterSpec);
            return this;
        }

        public IParameterSpec[] BuildParameterSpecs() => m_parameterSpecs.ToArray();
        public IOptimizerBuilder BuildOptimizer() => new OptimizerBuilder(BuildParameterSpecs());
    }
}
