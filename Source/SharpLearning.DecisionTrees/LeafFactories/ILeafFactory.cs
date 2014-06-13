
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Nodes;

namespace SharpLearning.DecisionTrees.LeafFactories
{
    /// <summary>
    /// Provides a leaf given a range of values and optional calculation interval
    /// </summary>
    /// <param name="values"></param>
    /// <param name="interval"></param>
    /// <returns></returns>
    public interface ILeafFactory
    {

        /// <summary>
        /// Provides a leaf given a range of values
        /// </summary>
        /// <param name="parent"></param>
        /// <param name="values"></param>
        /// <returns></returns>
        IBinaryDecisionNode Create(IBinaryDecisionNode parent, double[] values);

        /// <summary>
        /// Provides a leaf given a range of values and a calculation interval
        /// </summary>
        /// <param name="parent"></param>
        /// <param name="values"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        IBinaryDecisionNode Create(IBinaryDecisionNode parent, double[] values, Interval1D interval);
    }
}
