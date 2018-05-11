using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLearning.XGBoost.Learners
{
    internal static class ArgumentChecks
    {
        internal static void ThrowOnArgumentLessThan(string name, double value, double min)
        {
            if (value < min)
            {
                throw new ArgumentException($"{name} must be at least 0. was: {value}");
            }
        }

        internal static void ThrowOnArgumentLessThanOrHigherThan(string name, double value, 
            double min, double max)
        {
            if (value < min || value > max)
            {
                throw new ArgumentException($"{name} must be in range [{min};{max}]. Was: {value}");
            }
        }

    }
}
