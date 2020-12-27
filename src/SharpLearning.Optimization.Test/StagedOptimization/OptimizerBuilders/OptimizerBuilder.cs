using System;

namespace SharpLearning.Optimization.Test.StagedOptimization.OptimizerBuilders
{
    public class OptimizerBuilder : IOptimizerBuilder
    {
        readonly IParameterSpec[] m_parameterSpecs;
        CreateOptimizer m_createOptimizer;

        public OptimizerBuilder(IParameterSpec[] parameterSpecs)
        {
            m_parameterSpecs = parameterSpecs ?? throw new ArgumentNullException(nameof(parameterSpecs));
        }

        public IOptimizerBuilder AddOptimizer(CreateOptimizer createOptimizer)
        {
            m_createOptimizer = createOptimizer;
            return this;
        }

        public IOptimizer BuildOptimizer() => m_createOptimizer(m_parameterSpecs);
        public IRunOptimizerBuilder BuildObjectiveFunction() => new RunOptimizerBuilder(BuildOptimizer());
    }
}
