
namespace SharpLearning.Common.Interfaces
{
    /// <summary>
    /// General interface for predictor. 
    /// </summary>
    /// <typeparam name="TPrediction">The prediction type of the resulting model.</typeparam>
    public interface IPredictor<TPrediction>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        TPrediction Predict(double[] observation);
    }
}
