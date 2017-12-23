using System;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Delegate for acquisition functions.
    /// </summary>
    /// <param name="currentScore">Current best score.</param>
    /// <param name="mean">Predicted score.</param>
    /// <param name="variance">Predicted variance.</param>
    /// <param name="xi">Controls the balance between exploration and exploitation. Default is 0.0.</param>
    /// <returns>Expected Improvement.</returns>
    public delegate double AcquisitionFunction(double currentScore, double mean, double variance, double xi = 0.0);

    /// <summary>
    /// Acquisition functions for bayesian optimization
    /// </summary>
    public static class AcquisitionFunctions
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="currentScore">Current best score.</param>
        /// <param name="mean">Predicted score.</param>
        /// <param name="variance">Predicted variance.</param>
        /// <param name="xi">Controls the balance between exploration and exploitation. Default is 0.0.</param>
        /// <returns>Expected Improvement.</returns>
        public static double ExpectedImprovement(double currentScore, double mean, double variance, double xi = 0.0)
        {
            // in case of zero variance return 0.0.
            if (variance == 0.0) return 0.0;

            var std = Math.Sqrt(variance);
            var z = (currentScore - mean - xi) / std;
            var f = std * (z * CumulativeDensityFunction(z) + ProbabilityDensityFunction(z));

            return f;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="currentScore">Current best score.</param>
        /// <param name="mean">Predicted score.</param>
        /// <param name="variance">Predicted variance.</param>
        /// <param name="xi">Controls the balance between exploration and exploitation. Default is 0.0.</param>
        /// <returns>Probability of Improvement.</returns>
        public static double ProbabilityOfImprovement(double currentScore, double mean, double variance, double xi = 0.0)
        {
            // in case of zero variance return 0.0.
            if (variance == 0.0) return 0.0;

            var std = Math.Sqrt((double)variance);
            var z = (currentScore - mean - xi) / std;
            var f = CumulativeDensityFunction(z);

            return f;
        }

        static double ProbabilityDensityFunction(double x)
        {
            return Math.Exp(-x * x / 2.0) / Math.Sqrt(2.0 * Math.PI);
        }

        static double CumulativeDensityFunction(double x)
        {
            // constants
            double a1 = 0.254829592;
            double a2 = -0.284496736;
            double a3 = 1.421413741;
            double a4 = -1.453152027;
            double a5 = 1.061405429;
            double p = 0.3275911;

            // Save the sign of x
            int sign = 1;
            if (x < 0)
                sign = -1;
            x = Math.Abs(x) / Math.Sqrt(2.0);

            // A&S formula 7.1.26
            double t = 1.0 / (1.0 + p * x);
            double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

            return 0.5 * (1.0 + sign * y);
        }
    }
}
