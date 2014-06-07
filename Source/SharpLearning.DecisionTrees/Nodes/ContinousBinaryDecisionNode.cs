using System;

namespace SharpLearning.DecisionTrees.Nodes
{
    public sealed class ContinousBinaryDecisionNode : BinaryDicisionNodeBase, IBinaryDecisionNode
    {
        /// <summary>
        /// Predicts using a continous node strategy
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public override double Predict(double[] observation)
        {
            if (IsLeaf())
            {
                return Value;
            }

            if (observation[FeatureIndex] > Value)
            {
                return Left.Predict(observation);
            }
            else
            {
                return Right.Predict(observation);
            }

            throw new InvalidOperationException("The tree is degenerated.");
        }
    }
}
