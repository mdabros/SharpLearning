using SharpLearning.Containers.Matrices;

namespace SharpLearning.Common.Interfaces
{
    /// <summary>
    /// Interface for indexed learner. 
    /// Only the observations from the provided indices in the index array will be used for training
    /// </summary>
    /// <typeparam name="TPrediction">The prediction type of the resulting model.</typeparam>
    public interface IIndexedLearner<TPrediction>
    {
        /// <summary>
        /// Only the observations from the provided indices in the index array will be used for training
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictorModel<TPrediction> Learn(F64Matrix observations, double[] targets, int[] indices);
    }
}
