using System.Collections.Generic;

namespace SharpLearning.Optimization.Test.StagedOptimization
{
    /// <summary>
    /// Provides the main entry point and execution functionality
    /// </summary>
    /// <typeparam name="TControllableStep"></typeparam>
    public interface IControlFlowScheduler
    {
        /// <summary>
        /// Readies the control flow scheduler for sequence definition
        /// </summary>
        /// <returns></returns>
        IControlFlowDoer Initialize();

        /// <summary>
        /// executes the currently defined sequence 
        /// </summary>
        IDictionary<string, object> Execute();
    }
}
