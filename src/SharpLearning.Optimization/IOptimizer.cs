using System;
using System.Collections.Generic;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// 
    /// </summary>
    /// <returns></returns>
    public delegate OptimizerResult FunctionToMinimize(Dictionary<string, double> nameToParameter);

    /// <summary>
    /// 
    /// </summary>
    public interface IOptimizer
    {
        /// <summary>
        /// Returns the result which best minimizes the provided function.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        OptimizerResult OptimizeBest(FunctionToMinimize functionToMinimize);
        
        /// <summary>
        /// Returns all results ordered from best to worst (minimized). 
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        OptimizerResult[] Optimize(FunctionToMinimize functionToMinimize);
    }
}
