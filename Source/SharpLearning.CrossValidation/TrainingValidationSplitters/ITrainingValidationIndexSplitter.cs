
namespace SharpLearning.CrossValidation.TrainingValidationSplitters
{
    /// <summary>
    /// Interface for training validation index splitters
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface ITrainingValidationIndexSplitter<T>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="targets"></param>
        /// <returns></returns>
        TrainingValidationIndexSplit Split(T[] targets);
    }
}
