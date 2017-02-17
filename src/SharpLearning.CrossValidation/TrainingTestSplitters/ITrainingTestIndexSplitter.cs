
namespace SharpLearning.CrossValidation.TrainingTestSplitters
{
    /// <summary>
    /// Interface for training test index splitters
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface ITrainingTestIndexSplitter<T>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="targets"></param>
        /// <returns></returns>
        TrainingTestIndexSplit Split(T[] targets);
    }
}
