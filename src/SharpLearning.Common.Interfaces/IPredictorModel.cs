namespace SharpLearning.Common.Interfaces
{
    /// <summary>
    /// Interface for predictor models. Supports prediction and variable importance.
    /// </summary>
    /// <typeparam name="TPrediction"></typeparam>
    public interface IPredictorModel<TPrediction> : IPredictor<TPrediction>, IModelVariableImportance
    {
    }
}
