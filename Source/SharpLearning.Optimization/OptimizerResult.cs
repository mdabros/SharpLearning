using System;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Optimization result
    /// </summary>
    public sealed class OptimizerResult
    {
        /// <summary>
        /// Resulting error using the parameter set
        /// </summary>
        public readonly double Error;

        /// <summary>
        /// The parameter set
        /// </summary>
        public readonly double[] ParameterSet;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="error"></param>
        /// <param name="parameterSet"></param>
        public OptimizerResult(double[] parameterSet, double error)
        {
            if (parameterSet == null) { throw new ArgumentException("parameterSet"); }
            ParameterSet = parameterSet;
            Error = error;
        }
    }
}
