using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public class ControlFlowScheduler<TControllableStep> : IControlFlowScheduler<TControllableStep>
        where TControllableStep : IControlFlowStep
    {
        readonly Func<IControlFlowStepThenDoer<TControllableStep>> m_controlFlowStepGrouperFactory;
        IControlFlowStepThenDoer<TControllableStep> m_controlFlowStepThenDoer;
        IDictionary<Type, IControlFlowStep> m_controllableStepsDictionary;

        public ControlFlowScheduler(
            Func<IControlFlowStepThenDoer<TControllableStep>> controlFlowStepGrouperFactory,
            IEnumerable<TControllableStep> controllableSteps)
        {
            m_controlFlowStepGrouperFactory = controlFlowStepGrouperFactory;
            m_controllableStepsDictionary = new Dictionary<Type, IControlFlowStep>();
            foreach (var controllableStep in controllableSteps)
            {
                m_controllableStepsDictionary.Add(controllableStep.GetType(), controllableStep);
            }
        }

        public IControlFlowStepThenDoer<TControllableStep> Initialize()
        {
            m_controlFlowStepThenDoer = m_controlFlowStepGrouperFactory.Invoke();
            return m_controlFlowStepThenDoer;
        }

        public void Execute()
        { 
            foreach (IReadOnlyCollection<Type> groupsOfStepsToExecute in 
                ((ControlFlowDoer<TControllableStep>)(m_controlFlowStepThenDoer)).m_sequenceOfGroupsOfStepsToExecute)
            {
                if (groupsOfStepsToExecute.Count == 1)
                    ExecuteStep(groupsOfStepsToExecute.First());
                else
                    ExecuteSteps(groupsOfStepsToExecute);
            }
        }

        private void ExecuteStep(Type type)
        {
            IControlFlowStep sequentialStep = m_controllableStepsDictionary[type];
            sequentialStep.Execute();
        }

        private void ExecuteSteps(IEnumerable<Type> types)
        {
            Parallel.ForEach
            (
                types, async (type) => 
                {
                    IControlFlowStep aConcurrentStep = m_controllableStepsDictionary[type];
                    await aConcurrentStep.ExecuteAsync();
                }
            );
        }

    }
}
