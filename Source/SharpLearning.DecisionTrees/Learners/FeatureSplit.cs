
namespace SharpLearning.DecisionTrees.Learners
{
    /// <summary>
    /// Contains the the value and index created by a feature split
    /// </summary>
    public struct FeatureSplit
    {
        public readonly double Value;
        public readonly int Index;

        public FeatureSplit(double value, int index)
        {
            this.Value = value;
            this.Index = index;
        }
    }
}
