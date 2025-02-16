using SharpLearning.GradientBoost.GBMDecisionTree;

namespace SharpLearning.GradientBoost.Loss;

/// <summary>
/// Interface for gradient boost loss functions
/// </summary>
public interface IGradientBoostLoss
{
    /// <summary>
    /// Calculate the initial, constant, loss based on the targets and the samples used
    /// </summary>
    /// <param name="targets"></param>
    /// <param name="inSample"></param>
    /// <returns></returns>
    double InitialLoss(double[] targets, bool[] inSample);

    /// <summary>
    /// Initialize the split search based on targets, residuals and the samples used
    /// </summary>
    /// <param name="targets"></param>
    /// <param name="residuals"></param>
    /// <param name="inSample"></param>
    /// <returns></returns>
    GBMSplitInfo InitSplit(double[] targets, double[] residuals, bool[] inSample);

    /// <summary>
    /// Calculate the negative gradient
    /// </summary>
    /// <param name="target"></param>
    /// <param name="prediction"></param>
    /// <returns></returns>
    double NegativeGradient(double target, double prediction);

    /// <summary>
    /// Update the residuals using the negative gradient
    /// </summary>
    /// <param name="targets"></param>
    /// <param name="predictions"></param>
    /// <param name="residuals"></param>
    /// <param name="inSample"></param>
    void UpdateResiduals(double[] targets, double[] predictions, double[] residuals, bool[] inSample);

    /// <summary>
    /// Update left and right split values based on the target and residual
    /// </summary>
    /// <param name="left"></param>
    /// <param name="right"></param>
    /// <param name="target"></param>
    /// <param name="residual"></param>
    void UpdateSplitConstants(ref GBMSplitInfo left, ref GBMSplitInfo right,
        double target, double residual);

    /// <summary>
    /// Does the loss function need to update leaf values after the split has been found
    /// </summary>
    /// <returns></returns>
    bool UpdateLeafValues();

    /// <summary>
    /// Provides an updated leaf value based on the tagets and predictions and the samples used
    /// </summary>
    /// <param name="currentLeafValue"></param>
    /// <param name="targets"></param>
    /// <param name="predictions"></param>
    /// <param name="inSample"></param>
    /// <returns></returns>
    double UpdatedLeafValue(double currentLeafValue, double[] targets, double[] predictions, bool[] inSample);
}
