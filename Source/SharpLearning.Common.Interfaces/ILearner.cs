using SharpLearning.Containers.Matrices;

namespace SharpLearning.Common.Interfaces
{
    /// <summary>
    /// General interface for learner. 
    /// </summary>
    /// <typeparam name="TPrediction"></typeparam>
    public interface ILearner<TPrediction>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<TPrediction> Learn(F64Matrix observations, double[] targets);
    }
}
