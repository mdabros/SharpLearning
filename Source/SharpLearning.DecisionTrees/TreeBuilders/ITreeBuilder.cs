using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Nodes;

namespace SharpLearning.DecisionTrees.TreeBuilders
{
    /// <summary>
    /// Tree builder interface
    /// </summary>
    public interface ITreeBuilder
    {
        BinaryTree Build(F64MatrixView observations, double[] targets, int[] indices, double[] weights);
    }
}
