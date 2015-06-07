using SharpLearning.Containers.Extensions;
using System;

namespace SharpLearning.GradientBoost.GBM
{
    /// <summary>
    /// Binomial deviation is used for binary classification. It penalizes a misclassification more heavily than a correct classification
    /// which makes it possible to reduce the misclassification rate during a learning process.
    /// Its penalty increases linearly with f which makes it more robust to outliers
    /// than other loss functions for which the penalty increases at a higher rate
    /// </summary>
    public sealed class GBMBinomialLoss : IGBMLoss
    {
        /// Binomial deviation is used for binary classification. It penalizes a misclassification more heavily than a correct classification
        /// which makes it possible to reduce the misclassification rate during a learning process.
        /// Its penalty increases linearly with f which makes it more robust to outliers
        /// than other loss functions for which the penalty increases at a higher rate
        public GBMBinomialLoss()
        {
        }

        /// <summary>
        /// initial loss is the class prior probability
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

            return Math.Log(inSampleSum / (sampleCount - inSampleSum)).NanToNum();
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
                    var target = targets[i];
                    var residual = residuals[i];
                    var residual2 = residual * residual;
                    var binomial = (target - residual) * (1.0 - target + residual);

                    splitInfo.Samples++;
                    splitInfo.Sum += residual;
                    splitInfo.SumOfSquares += residual2;
                    splitInfo.BinomialSum += binomial;
                }
            }

            splitInfo.Cost = splitInfo.SumOfSquares - (splitInfo.Sum * splitInfo.Sum / (double)splitInfo.Samples);
            splitInfo.BestConstant = BinomialBestConstant(splitInfo.Sum, splitInfo.BinomialSum);

            return splitInfo;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="target"></param>
        /// <param name="prediction"></param>
        /// <returns></returns>
        public double NegativeGradient(double target, double prediction)
        {
            return (target - 1.0 / (1.0 + Math.Exp(-prediction))).NanToNum();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="residuals"></param>
        public void UpdateResiduals(double[] targets, double[] predictions, double[] residuals, bool[] inSample)
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
            var binomial = (target - residual) * (1.0 - target + residual);

            left.Samples++;
            left.Sum += residual;
            left.SumOfSquares += residual2;
            left.BinomialSum += binomial;
            left.Cost = left.SumOfSquares - (left.Sum * left.Sum / (double)left.Samples);
            left.BestConstant = BinomialBestConstant(left.Sum, left.BinomialSum);

            right.Samples--;
            right.Sum -= residual;
            right.SumOfSquares -= residual2;
            right.BinomialSum -= binomial;
            right.Cost = right.SumOfSquares - (right.Sum * right.Sum / (double)right.Samples);
            right.BestConstant = BinomialBestConstant(right.Sum, right.BinomialSum);
        }

        double BinomialBestConstant(double sum, double binomialSum)
        {
            if (binomialSum != 0.0)
            {
                return sum / binomialSum;
            }
            else
            {
                return 0.0;
            }
        }

        /// <summary>
        /// Binomial loss does not require leaf value updates
        /// </summary>
        /// <param name="currentLeafValue"></param>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="inSample"></param>
        /// <returns></returns>
        public double UpdatedLeafValue(double currentLeafValue, double[] targets, double[] predictions, bool[] inSample)
        {
            // no update needed for binomial loss
            return currentLeafValue;
        }

        /// <summary>
        /// Binomial loss does not require leaf value updates
        /// </summary>
        /// <returns></returns>
        public bool UpdateLeafValues()
        {
            return false;
        }
    }
}
