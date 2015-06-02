using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.GradientBoost.GBM
{
    /// <summary>
    /// The square loss function is the standard method of fitting regression models.
    /// The square loss is however sensitive to outliers since it weighs larger errors more heavily than small ones.
    /// In case of many outliers Least absolute deviation (LAD) is a better alternative.
    /// </summary>
    public sealed class GBMSquaredLoss : IGBMLoss
    {
        /// <summary>
        /// The square loss function is the standard method of fitting regression models.
        /// The square loss is however sensitive to outliers since it weighs larger errors more heavily than small ones.
        /// In case of many outliers Least absolute deviation (LAD) is a better alternative.
        /// </summary>
        public GBMSquaredLoss()
        {

        }

        /// <summary>
        /// Initial loss is the mean of the targets 
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="inSample"></param>
        /// <returns></returns>
        public double InitialLoss(double[] targets, bool[] inSample)
        {
            var inSampleSum = 0.0;
            var sampleCount = 0.0;
            for (int i = 0; i < inSample.Length; i++)
            {
                if (inSample[i])
                {
                    inSampleSum += targets[i];
                    sampleCount++;
                }
            }

            return inSampleSum / sampleCount;
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
            splitInfo.BestConstant = splitInfo.Sum / (double)splitInfo.Samples;

            return splitInfo;
        }

        /// <summary>
        /// Negative gradient is the difference between target and prediction
        /// </summary>
        /// <param name="target"></param>
        /// <param name="prediction"></param>
        /// <returns></returns>
        public double NegativeGradient(double target, double prediction)
        {
            return target - prediction;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="residuals"></param>
        public void UpdateResiduals(double[] targets, double[] predictions, double[] residuals)
        {
            for (int i = 0; i < residuals.Length; i++)
            {
                residuals[i] = NegativeGradient(targets[i], predictions[i]);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="target"></param>
        /// <param name="residual"></param>
        public void UpdateSplitConstants(GBMSplitInfo left, GBMSplitInfo right, double target, double residual)
        {
            var residual2 = residual * residual;

            left.Samples++;
            left.Sum += residual;
            left.SumOfSquares += residual2;
            left.Cost = left.SumOfSquares - (left.Sum * left.Sum / (double)left.Samples);
            left.BestConstant = left.Sum / (double)left.Samples;

            right.Samples--;
            right.Sum -= residual;
            right.SumOfSquares -= residual2;
            right.Cost = right.SumOfSquares - (right.Sum * right.Sum / (double)right.Samples);
            right.BestConstant = right.Sum / (double)right.Samples;
        }

        /// <summary>
        /// Squared loss does not require to update the leaf values
        /// </summary>
        /// <param name="currentLeafValue"></param>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="inSample"></param>
        /// <returns></returns>
        public double UpdatedLeafValue(double currentLeafValue, double[] targets, double[] predictions, bool[] inSample)
        {
            // no updates needed for square loss
            return currentLeafValue;
        }

        /// <summary>
        /// Squared loss does not require to update the leaf values
        /// </summary>
        /// <returns></returns>
        public bool UpdateLeafValues()
        {
            return false;
        }
    }
}
