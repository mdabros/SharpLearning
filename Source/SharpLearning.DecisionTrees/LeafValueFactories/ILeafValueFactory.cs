
using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.LeafValueFactories
{
    /// <summary>
    /// Provides the value of a leaf given a range of values and a calculation interval
    /// </summary>
    /// <param name="values"></param>
    /// <param name="interval"></param>
    /// <returns></returns>
    public interface ILeafValueFactory
    {

        /// <summary>
        /// Provides the value of a leaf given a range of values
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        double Calculate(double[] values);

        /// <summary>
        /// Provides the value of a leaf given a range of values and a calculation interval
        /// </summary>
        /// <param name="values"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        double Calculate(double[] values, Interval1D interval);
    }
}
