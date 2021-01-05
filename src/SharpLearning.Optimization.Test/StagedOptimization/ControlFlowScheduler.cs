using System;

namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public class ControlFlowScheduler : IControlFlowScheduler
    {
        readonly Func<IControlFlowDoer> m_controlFlowFactory;
        IControlFlowDoer m_controlFlowDoer;

        public ControlFlowScheduler(
            Func<IControlFlowDoer> controlFlowFactory)
        {
            m_controlFlowFactory = controlFlowFactory;
        }

        public IControlFlowDoer Initialize()
        {
            m_controlFlowDoer = m_controlFlowFactory.Invoke();
            return m_controlFlowDoer;
        }

        public IRepository Execute()
        {
            var stagesRepository = new Repository();

            foreach (var stageStep in ((ControlFlowDoer)m_controlFlowDoer).Steps())
            {
                stageStep(stagesRepository);
            }

            return stagesRepository;
        }
    }
}
