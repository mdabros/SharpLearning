using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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

        public IDictionary<string, object> Execute()
        {
            var stagesRepository = new Dictionary<string, object>();

            foreach (var stageStep in ((ControlFlowDoer)m_controlFlowDoer).Steps())
            {
                stageStep(stagesRepository);
            }

            return stagesRepository;
        }
    }
}
