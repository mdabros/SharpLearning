using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Nodes;

namespace SharpLearning.DecisionTrees.TreeBuilders
{
    /// <summary>
    /// Tree builder interface
    /// </summary>
    public interface ITreeBuilder
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        BinaryTree Build(F64MatrixView observations, double[] targets, int[] indices, double[] weights);
    }
}
