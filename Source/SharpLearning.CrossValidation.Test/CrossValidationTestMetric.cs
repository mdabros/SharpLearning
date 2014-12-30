using SharpLearning.Learners.Interfaces;
using System;

namespace SharpLearning.CrossValidation.Test
{
    internal sealed class CrossValidationTestMetric : IMetric<double, double>
    {
        public double Error(double[] targets, double[] predictions)
        {
            if (targets.Length != predictions.Length) { throw new ArgumentException("targets and predictions length do not match"); }

            var meanSquareError = 0.0;
            for (int i = 0; i < targets.Length; ++i)
            {
                var targetValue = targets[i];
                var estimate = predictions[i];
                var error = targetValue - estimate;
                meanSquareError += error * error;
            }
            meanSquareError *= (1.0 / targets.Length);

            return meanSquareError;
        }
    }
}
