
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Common.Interfaces
{
    /// <summary>
    /// General interface for predictor. 
    /// </summary>
    /// <typeparam name="TPrediction">The prediction type of the resulting model.</typeparam>
    public interface IPredictor<TPrediction>
    {
        /// <summary>
        /// Predicts a single observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        TPrediction Predict(double[] observation);

        /// <summary>
        /// Predicts a set of observations
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        TPrediction[] Predict(F64Matrix observations);
    }
}
