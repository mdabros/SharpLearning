using System;
using System.Collections.Generic;

namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public class ControlFlowDoer : IControlFlowDoer
    {
        internal protected List<StageStep> m_steps = new List<StageStep>();

        public IControlFlowDoer Do(StageStep step) 
        {
            m_steps.Add(step);
            return this;
        }

        public IReadOnlyList<StageStep> Steps() => m_steps;
    }
}
