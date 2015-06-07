using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.GBMDecisionTree;

namespace SharpLearning.GradientBoost.LossFunctions
{
    /// <summary>
    /// Least absolute deviation (LAD) loss function. LAD gives equal equal emphasis to all observations. 
    /// This makes LAD robust against outliers.
    /// http://en.wikipedia.org/wiki/Least_absolute_deviations
    /// </summary>
    public sealed class GBMAbsoluteLoss : IGBMLoss
    {
        /// <summary>
        /// Least absolute deviation (LAD) loss function. LAD gives equal equal emphasis to all observations. 
        /// This makes LAD robust against outliers. LAD regression is also sometimes known as robust regression. 
        /// http://en.wikipedia.org/wiki/Least_absolute_deviations
        /// </summary>
        public GBMAbsoluteLoss()
        {
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
            for (int i = 0; i < inSample.Length; i++)
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
        /// Negative gradient is either 1 or -1 depending on the sign of target minus prediction
        /// </summary>
        /// <param name="target"></param>
        /// <param name="prediction"></param>
        /// <returns></returns>
        public double NegativeGradient(double target, double prediction)
        {
            var value = target - prediction;
            if (value > 0.0)
            {
                return 1.0;
            }
            else
            {
                return -1.0;
            }
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
        /// Leaf values are updated using the median of the difference between target and prediction
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

            return values.ToArray().Median();
        }

        public bool UpdateLeafValues()
        {
            return true;
        }
    }
}
