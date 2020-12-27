using System;
using System.Collections.Generic;

namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public class ControlFlowDoer<TControllableStep> : IControlFlowDoer<TControllableStep>
        where TControllableStep : IControlFlowStep
    {
        private protected ControlFlowThenDoer<TControllableStep> m_controlFlowThenDoer;
        internal protected List<Type> m_mostRecentCollectionOfConcurrentSteps;
        internal protected List<List<Type>> m_sequenceOfGroupsOfStepsToExecute = new List<List<Type>>();

        public IControlFlowStepThenDoer<TControllableStep> Do<TControllableStepAlias>() 
            where TControllableStepAlias : TControllableStep
        {
            m_mostRecentCollectionOfConcurrentSteps
                .Add(typeof(TControllableStepAlias));
            return m_controlFlowThenDoer;
        }
    }
}
