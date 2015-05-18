
namespace SharpLearning.GradientBoost.GBM
{
    public class GBMSplit
    {
        public int Depth;
        public int FeatureIndex;
        public int SplitIndex;
        public double SplitValue;
        public double LeftError;
        public double RightError;
        public double LeftConstant;
        public double RightConstant;
        public double Cost;
        public double CostImprovement;

        public GBMNode GetNode()
        {
            return new GBMNode
            {
                FeatureIndex = FeatureIndex,
                SplitValue = SplitValue,
                LeftError = LeftError,
                RightError = RightError,
                LeftConstant = LeftConstant,
                RightConstant = RightConstant,
                Depth = Depth,
            };
        }
    }
}
