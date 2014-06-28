
namespace SharpLearning.CrossValidation.Shufflers
{
    /// <summary>
    /// Shuffles the provided indices based on the targets and folds
    /// </summary>
    public interface ICrossValidationShuffler<T>
    {
        /// <summary>
        /// Shuffles the provided indices based on the targets and folds
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="targets"></param>
        /// <param name="folds"></param>
        void Shuffle(int[] indices, T[] targets, int folds);
    }
}
