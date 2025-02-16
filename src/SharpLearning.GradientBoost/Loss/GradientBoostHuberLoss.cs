using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.GBMDecisionTree;

namespace SharpLearning.GradientBoost.Loss;

/// <summary>
/// Huber loss is a combination of Squared loss and least absolute deviation (LAD).
/// For small residuals (below quantile defined by alpha) squared loss is used.
/// For large residuals (above quantile defined by alpha) LAD loss is used.
/// This makes Huber loss robust against outliers while still having much of the sensitivity of squared loss.
/// http://en.wikipedia.org/wiki/Huber_loss
/// </summary>
public sealed class GradientBoostHuberLoss : IGradientBoostLoss
{
    double m_gamma;
    readonly double m_alpha;

    /// Huber loss is a combination of Squared loss and least absolute deviation (LAD).
    /// For small residuals (below quantile defined by alpha) squared loss is used.
    /// For large residuals (above quantile defined by alpha) LAD loss is used.
    /// This makes Huber loss robust against outliers while still having much of the sensitivity of squared loss.
    /// http://en.wikipedia.org/wiki/Huber_loss
    public GradientBoostHuberLoss(double alpha = 0.9)
    {
        if (alpha <= 0.0 || alpha > 1.0) { throw new ArgumentException("Alpha must be larger than 0.0 and no more than 1.0"); }
        m_alpha = alpha;
    }

    /// <summary>
    /// Initial loss is the median
    /// </summary>
    /// <param name="targets"></param>
    /// <param name="inSample"></param>
    /// <returns></returns>
    public double InitialLoss(double[] targets, bool[] inSample)
    {
        var values = new List<double>();
        for (var i = 0; i < inSample.Length; i++)
        {
            if (inSample[i])
            {
                values.Add(targets[i]);
            }
        }

        return values.ToArray().Median();
    }

    /// <summary>
    ///
    /// </summary>
    /// <param name="targets"></param>
    /// <param name="residuals"></param>
    /// <param name="inSample"></param>
    /// <returns></returns>
    public GBMSplitInfo InitSplit(double[] targets, double[] residuals, bool[] inSample)
    {
        var splitInfo = GBMSplitInfo.NewEmpty();

        for (var i = 0; i < inSample.Length; i++)
        {
            if (inSample[i])
            {
                var residual = residuals[i];
                var residual2 = residual * residual;

                splitInfo.Samples++;
                splitInfo.Sum += residual;
                splitInfo.SumOfSquares += residual2;
            }
        }

        splitInfo.Cost = splitInfo.SumOfSquares - (splitInfo.Sum * splitInfo.Sum / splitInfo.Samples);

        return splitInfo;
    }

    /// <summary>
    /// Undefined for Huber
    /// </summary>
    /// <param name="target"></param>
    /// <param name="prediction"></param>
    /// <returns></returns>
    public double NegativeGradient(double target, double prediction)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    ///
    /// </summary>
    /// <param name="targets"></param>
    /// <param name="predictions"></param>
    /// <param name="residuals"></param>
    /// <param name="inSample"></param>
    public void UpdateResiduals(double[] targets, double[] predictions,
        double[] residuals, bool[] inSample)
    {
        var absDiff = new double[inSample.Length];
        var difference = new double[inSample.Length];

        for (var i = 0; i < inSample.Length; i++)
        {
            if (inSample[i])
            {
                var value = targets[i] - predictions[i];
                difference[i] = value;
                absDiff[i] = Math.Abs(value);
            }
        }

        var gamma = absDiff.ToArray().ScoreAtPercentile(m_alpha);

        for (var i = 0; i < inSample.Length; i++)
        {
            if (inSample[i])
            {
                var diff = absDiff[i];

                if (diff <= gamma)
                {
                    residuals[i] = difference[i];
                }
                else
                {
                    residuals[i] = gamma * Math.Sign(difference[i]);
                }
            }
        }

        m_gamma = gamma;
    }

    /// <summary>
    ///
    /// </summary>
    /// <param name="left"></param>
    /// <param name="right"></param>
    /// <param name="target"></param>
    /// <param name="residual"></param>
    public void UpdateSplitConstants(ref GBMSplitInfo left, ref GBMSplitInfo right,
        double target, double residual)
    {
        var residual2 = residual * residual;

        left.Samples++;
        left.Sum += residual;
        left.SumOfSquares += residual2;
        left.Cost = left.SumOfSquares - (left.Sum * left.Sum / left.Samples);

        right.Samples--;
        right.Sum -= residual;
        right.SumOfSquares -= residual2;
        right.Cost = right.SumOfSquares - (right.Sum * right.Sum / right.Samples);
    }

    /// <summary>
    ///
    /// </summary>
    /// <returns></returns>
    public bool UpdateLeafValues()
    {
        return true;
    }

    /// <summary>
    ///
    /// </summary>
    /// <param name="currentLeafValue"></param>
    /// <param name="targets"></param>
    /// <param name="predictions"></param>
    /// <param name="inSample"></param>
    /// <returns></returns>
    public double UpdatedLeafValue(double currentLeafValue, double[] targets,
        double[] predictions, bool[] inSample)
    {
        var diff = new List<double>();
        for (var j = 0; j < inSample.Length; j++)
        {
            if (inSample[j])
            {
                diff.Add(targets[j] - predictions[j]);
            }
        }

        var median = diff.ToArray().Median();
        var values = new double[diff.Count];

        for (var j = 0; j < diff.Count; j++)
        {
            var medianDiff = diff[j] - median;
            var sign = Math.Sign(medianDiff);

            values[j] = sign * Math.Min(Math.Abs(medianDiff), m_gamma);
        }

        var newValue = median + values.Sum() / values.Length;

        return newValue;
    }
}
