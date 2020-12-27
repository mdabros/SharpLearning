namespace SharpLearning.Optimization.Test.StagedOptimization
{
    /// <summary>
    /// Provides functionality to group or separate the execution 
    /// of ControlFlowSteps as per to the schedule
    /// </summary>
    /// <typeparam name="TControllableStep"></typeparam>
    public interface IControlFlowStepThenDoer<TControllableStep> : IControlFlowDoer<TControllableStep>
        where TControllableStep : IControlFlowStep
    {
        /// <summary>
        /// Adds a boundary between IControlFlowSteps that must be executed in sequentially
        /// </summary>
        /// <returns></returns>
        IControlFlowDoer<TControllableStep> Then();
    }
}
