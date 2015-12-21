using System;

namespace SharpLearning.Optimization
{
    public interface IOptimizer
    {
        OptimizerResult Optimize(Func<double[], OptimizerResult> functionToMinimize);
    }
}
