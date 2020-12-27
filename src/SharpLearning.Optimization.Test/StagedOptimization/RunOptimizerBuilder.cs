using System;

namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public class RunOptimizerBuilder : IRunOptimizerBuilder
    {
        readonly IOptimizer m_optimizer;
        ObjectiveFunction m_objectiveFunction;

        public RunOptimizerBuilder(IOptimizer optimizer)
        {
            m_optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        }

        public IRunOptimizerBuilder AddObjective(ObjectiveFunction objectiveFunction)
        {
            m_objectiveFunction = objectiveFunction;
            return this;
        }

        public RunOptimizer BuildRunOptimizer() =>
            () => m_optimizer.Optimize(p => m_objectiveFunction(p));
    }
}
