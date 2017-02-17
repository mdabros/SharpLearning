using System;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// 
    /// </summary>
    public interface IOptimizer
    {
        /// <summary>
        /// Returns the result which best minimises the provided function.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        OptimizerResult OptimizeBest(Func<double[], OptimizerResult> functionToMinimize);
        
        /// <summary>
        /// Returns all results ordered from best to worst (minimized). 
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        OptimizerResult[] Optimize(Func<double[], OptimizerResult> functionToMinimize);
    }
}
