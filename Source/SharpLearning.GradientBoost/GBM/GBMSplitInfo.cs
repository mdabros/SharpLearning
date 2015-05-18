using SharpLearning.DecisionTrees.Nodes;

namespace SharpLearning.GradientBoost.GBM
{
    public class GBMSplitInfo
    {
        public int Samples;
        public double Sum;
        public double SumOfSquares;
        public double Cost;
        public double BestConstant;
        public NodePositionType Position;

        public static GBMSplitInfo NewEmpty()
        {
            return new GBMSplitInfo
            {
                Samples = 0,
                Sum = 0,
                SumOfSquares = 0,
                Cost = 0,
                BestConstant = 0
            };
        }

        public GBMSplitInfo Copy()
        {
            return Copy(NodePositionType.Root);
        }

        public GBMSplitInfo Copy(NodePositionType Position)
        {
            return new GBMSplitInfo
            {
                Samples = Samples,
                Sum = Sum,
                SumOfSquares = SumOfSquares,
                Cost = Cost,
                BestConstant = BestConstant,
                Position = Position
            };
        }
    }
}
