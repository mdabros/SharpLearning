using SharpLearning.DecisionTrees.Nodes;

namespace SharpLearning.GradientBoost.GBMDecisionTree
{
    /// <summary>
    /// Contains information about the current split values
    /// </summary>
    public struct GBMSplitInfo
    {
        /// <summary>
        /// Number of samples in the split
        /// </summary>
        public int Samples;

        /// <summary>
        /// Current sum of the split
        /// </summary>
        public double Sum;

        /// <summary>
        /// Current sum of squares of the split
        /// </summary>
        public double SumOfSquares;
        
        /// <summary>
        /// Current cost of the split
        /// </summary>
        public double Cost;

        /// <summary>
        /// Current best constant (fitted value) of the split
        /// </summary>
        public double BestConstant;

        /// <summary>
        /// Binomial sum of the split
        /// </summary>
        public double BinomialSum;

        /// <summary>
        /// The node position of the split
        /// </summary>
        public NodePositionType Position;

        /// <summary>
        /// Creates a new empty split info with initial default values
        /// </summary>
        /// <returns></returns>
        public static GBMSplitInfo NewEmpty()
        {
            return new GBMSplitInfo
            {
                Samples = 0,
                Sum = 0,
                SumOfSquares = 0,
                Cost = 0,
                BestConstant = 0,
                BinomialSum = 0
            };
        }

        /// <summary>
        /// Creates a copy of the split info
        /// </summary>
        /// <returns></returns>
        public GBMSplitInfo Copy()
        {
            return Copy(NodePositionType.Root);
        }

        /// <summary>
        /// Creates a copy of the split info
        /// </summary>
        /// <param name="Position"></param>
        /// <returns></returns>
        public GBMSplitInfo Copy(NodePositionType Position)
        {
            return new GBMSplitInfo
            {
                Samples = Samples,
                Sum = Sum,
                SumOfSquares = SumOfSquares,
                Cost = Cost,
                BestConstant = BestConstant,
                Position = Position,
                BinomialSum = BinomialSum
            };
        }
    }
}
