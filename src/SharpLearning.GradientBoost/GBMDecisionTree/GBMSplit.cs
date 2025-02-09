
namespace SharpLearning.GradientBoost.GBMDecisionTree;

/// <summary>
/// Represents the a split when learning a gradient boost decision tree
/// </summary>
public struct GBMSplit
{
    /// <summary>
    /// Index of the feature that the node splits on
    /// </summary>
    public int FeatureIndex;

    /// <summary>
    /// Index of the split value
    /// </summary>
    public int SplitIndex;

    /// <summary>
    /// Value of the feature that the node splits on
    /// </summary>
    public double SplitValue;

    /// <summary>
    /// The error on the left side of the split
    /// </summary>
    public double LeftError;

    /// <summary>
    /// The error on the right side of the split
    /// </summary>
    public double RightError;

    /// <summary>
    /// Left constant (fitted value) of the split
    /// </summary>
    public double LeftConstant;

    /// <summary>
    /// Right constant (fitted value) of the split
    /// </summary>
    public double RightConstant;

    /// <summary>
    /// Depth of the node in the decision tree
    /// </summary>
    public int Depth;

    /// <summary>
    /// The number of observations in the node
    /// </summary>
    public int SampleCount;

    /// <summary>
    /// Cost of the split
    /// </summary>
    public double Cost;

    /// <summary>
    /// Cost improvement of the split compared to parent split
    /// </summary>
    public double CostImprovement;

    /// <summary>
    /// Creates a GBMNode from the split
    /// </summary>
    /// <returns></returns>
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
            SampleCount = SampleCount
        };
    }
}
