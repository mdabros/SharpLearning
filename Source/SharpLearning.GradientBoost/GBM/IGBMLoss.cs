using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.GradientBoost.GBM
{
    public interface IGBMLoss
    {
        double NegativeGradient(double target, double prediction);
        
        void UpdateSplitConstants(GBMSplitInfo left, GBMSplitInfo right, 
            double target, double residual);
    }

    public sealed class GBMSquaredLoss : IGBMLoss
    {
        public double NegativeGradient(double target, double prediction)
        {
            return target - prediction;
        }

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
    }

}
