using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.GBMDecisionTree;
using System;
using System.Collections.Generic;

namespace SharpLearning.GradientBoost.Loss
{
    /// <summary>
    /// Quantile loss. Whereas the method of least squares results in estimates that approximate the conditional mean of the response variable 
    /// given certain values of the predictor variables, quantile regression aims at estimating either the conditional median 
    /// or other quantiles of the response variable. Using the median results in Least absolute deviation or LAD loss.
    /// </summary>
    public sealed class GradientBoostQuantileLoss : IGradientBoostLoss
    {
        readonly double m_alpha;

        /// <summary>
        /// Quantile loss. Whereas the method of least squares results in estimates that approximate the conditional mean of the response variable 
        /// given certain values of the predictor variables, quantile regression aims at estimating either the conditional median 
        /// or other quantiles of the response variable. Using the median results in Least absolute deviation or LAD loss.
        /// </summary>
        /// <param name="alpha"></param>
        public GradientBoostQuantileLoss(double alpha)
        {
            if (alpha <= 0.0 || alpha > 1.0) { throw new ArgumentException("Alpha must larger than 0.0 and at most 1.0"); }
            m_alpha = alpha;
        }

        /// <summary>
        /// The specified quantile of the targets.
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="inSample"></param>
        /// <returns></returns>
        public double InitialLoss(double[] targets, bool[] inSample)
        {
            var values = new List<double>();
            for (int i = 0; i < inSample.Length; i++)
            {
                if (inSample[i])
                {
                    values.Add(targets[i]);
                }
            }

            return values.ToArray().ScoreAtPercentile(m_alpha);
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

            for (int i = 0; i < inSample.Length; i++)
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

            splitInfo.Cost = splitInfo.SumOfSquares - (splitInfo.Sum * splitInfo.Sum / (double)splitInfo.Samples);

            return splitInfo;
        }

        /// <summary>
        /// Negative gradient is the quantile or -(1.0 - quantile) depending on which is larger. Prediction or target.
        /// </summary>
        /// <param name="target"></param>
        /// <param name="prediction"></param>
        /// <returns></returns>
        public double NegativeGradient(double target, double prediction)
        {
            if (target > prediction)
            {
                return m_alpha;
            }
            else
            {
                return -(1.0 - m_alpha);
            }

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="residuals"></param>
        /// <param name="inSample"></param>
        public void UpdateResiduals(double[] targets, double[] predictions, double[] residuals, bool[] inSample)
        {
            for (int i = 0; i < residuals.Length; i++)
            {
                if (inSample[i])
                {
                    residuals[i] = NegativeGradient(targets[i], predictions[i]);
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="target"></param>
        /// <param name="residual"></param>
        public void UpdateSplitConstants(ref GBMSplitInfo left, ref GBMSplitInfo right, double target, double residual)
        {
            var residual2 = residual * residual;

            left.Samples++;
            left.Sum += residual;
            left.SumOfSquares += residual2;
            left.Cost = left.SumOfSquares - (left.Sum * left.Sum / (double)left.Samples);

            right.Samples--;
            right.Sum -= residual;
            right.SumOfSquares -= residual2;
            right.Cost = right.SumOfSquares - (right.Sum * right.Sum / (double)right.Samples);
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
        /// Updates the leaf values based on the quantile of the difference between target and prediction
        /// </summary>
        /// <param name="currentLeafValue"></param>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="inSample"></param>
        /// <returns></returns>
        public double UpdatedLeafValue(double currentLeafValue, double[] targets, double[] predictions, bool[] inSample)
        {
            var values = new List<double>();

            for (int i = 0; i < inSample.Length; i++)
            {
                if (inSample[i])
                {
                    values.Add(targets[i] - predictions[i]);
                }
            }

            return values.ToArray().ScoreAtPercentile(m_alpha);
        }
    }

}
