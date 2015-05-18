
namespace SharpLearning.GradientBoost.GBM
{
    public class GBMTreeCreationItem
    {
        public GBMSplitInfo Values;
        public bool[] InSample;
        public int Depth;
        public GBMNode Parent;
    }
}
