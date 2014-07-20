using SharpLearning.Containers;
using System;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Nodes
{
    public sealed class LeafBinaryDecisionNode : BinaryDicisionNodeBase, IBinaryDecisionNode
    {
        public readonly Dictionary<double, double> m_probabilities;

        public LeafBinaryDecisionNode(Dictionary<double, double> probabilities)
        {
            if (probabilities == null) { throw new ArgumentException("probabilities"); }
            m_probabilities = probabilities;
        }

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

            if (observation[FeatureIndex] <= Value)
            {
                return Left.Predict(observation);
            }
            else
            {
                return Right.Predict(observation);
            }

            throw new InvalidOperationException("The tree is degenerated.");
        }

        /// <summary>
        /// Predict probabilities using a continous node strategy
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public override ProbabilityPrediction PredictProbability(double[] observation)
        {
            if (IsLeaf())
            {
                return new ProbabilityPrediction(Value, m_probabilities);
            }

            if (observation[FeatureIndex] <= Value)
            {
                return Left.PredictProbability(observation);
            }
            else
            {
                return Right.PredictProbability(observation);
            }

            throw new InvalidOperationException("The tree is degenerated.");
        }
    }
}
