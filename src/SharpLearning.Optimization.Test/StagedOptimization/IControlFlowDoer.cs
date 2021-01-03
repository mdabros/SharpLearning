namespace SharpLearning.Optimization.Test.StagedOptimization
{
    /// <summary>
    /// Provides functionality to schedule execution of a control flow step.
    /// </summary>
    /// <typeparam name="TControllableStep"></typeparam>
    public interface IControlFlowDoer
    {
        /// <summary>
        /// Schedules the IControlFlowStep type parameter for execution
        /// </summary>
        /// <returns></returns>
        IControlFlowDoer Do(StageStep step);
    }

}