namespace SharpLearning.Optimization.Test.StagedOptimization
{
    /// <summary>
    /// Provides the main entry point and execution functionality
    /// </summary>
    /// <typeparam name="TControllableStep"></typeparam>
    public interface IControlFlowScheduler<TControllableStep>
        where TControllableStep : IControlFlowStep
    {
        /// <summary>
        /// Readies the control flow scheduler for sequence definition
        /// </summary>
        /// <returns></returns>
        IControlFlowStepThenDoer<TControllableStep> Initialize();

        /// <summary>
        /// executes the currently defined sequence 
        /// </summary>
        void Execute();
    }
}
