using System;
using System.Collections.Generic;

namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public class ControlFlowThenDoer<TControllableStep> :
        ControlFlowDoer<TControllableStep>, IControlFlowStepThenDoer<TControllableStep>
        where TControllableStep : IControlFlowStep
    {
        public ControlFlowThenDoer() 
        {
            m_controlFlowThenDoer = this;
        }

        public IControlFlowDoer<TControllableStep> Then()
        {
            m_mostRecentCollectionOfConcurrentSteps = new List<Type>();
            m_sequenceOfGroupsOfStepsToExecute.Add(m_mostRecentCollectionOfConcurrentSteps);
            return this;
        }
    }
}
