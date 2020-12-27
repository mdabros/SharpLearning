namespace SharpLearning.Optimization.Test.StagedOptimization
{
    /// <summary>
    /// Provides functionality to schedule execution of a control flow step.
    /// </summary>
    /// <typeparam name="TControllableStep"></typeparam>
    public interface IControlFlowDoer<TControllableStep>
        where TControllableStep : IControlFlowStep
    {
        /// <summary>
        /// Schedules the IControlFlowStep type parameter for execution
        /// </summary>
        /// <typeparam name="TControllableStepAlias"></typeparam>
        /// <returns></returns>
        IControlFlowStepThenDoer<TControllableStep> Do<TControllableStepAlias>()
            where TControllableStepAlias : TControllableStep;
    }

}