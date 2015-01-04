
namespace SharpLearning.CrossValidation.Shufflers
{
    /// <summary>
    /// Does not shuffle the indices and keeps the original order.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class NoShuffleCrossValidationShuffler<T> : ICrossValidationShuffler<T>
    {
        /// <summary>
        /// Does not shuffle the indices and keeps the original order.
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="targets"></param>
        /// <param name="folds"></param>
        public void Shuffle(int[] indices, T[] targets, int folds)
        {
            // no shuffle.
        }
    }
}
