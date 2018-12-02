using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Simple grid search that tries all combinations of the provided parameters
    /// </summary>
    public sealed class GridSearchOptimizer : IOptimizer
    {
        readonly bool m_runParallel;
        readonly GridParameter[] m_parameterBounds;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parameterBounds">Each row is a series of values for a specific parameter</param>
        /// <param name="runParallel">Use multi threading to speed up execution (default is true)</param>
        public GridSearchOptimizer(GridParameter[] parameterBounds, bool runParallel = true)
        {
            if (parameterBounds == null) { throw new ArgumentNullException("parameterRanges"); }
            m_parameterBounds = parameterBounds;
            m_runParallel = runParallel;
        }

        /// <summary>
        /// Simple grid search that tries all combinations of the provided parameters.
        /// Returns the result which best minimises the provided function.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult OptimizeBest(Func<double[], OptimizerResult> functionToMinimize)
        {
            // Return the best model found.
            return Optimize(functionToMinimize).First();
        }

        /// <summary>
        /// Simple grid search that tries all combinations of the provided parameters.
        /// Returns all results ordered from best to worst (minimized).
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult[] Optimize(Func<double[], OptimizerResult> functionToMinimize)
        {
            // Generate the cartesian product between all parameters
            var grid = CartesianProduct(m_parameterBounds);

            // Initialize the search
            var results = new ConcurrentBag<OptimizerResult>();
            if (!m_runParallel)
            {
                foreach (var param in grid)
                {
                    // Get the current parameters for the current point
                    var result = functionToMinimize(param);
                    results.Add(result);
                }
            }
            else
            {
                var rangePartitioner = Partitioner.Create(grid, true);
                Parallel.ForEach(rangePartitioner, (param, loopState) =>
                {
                    // Get the current parameters for the current point
                    var result = functionToMinimize(param);
                    results.Add(result);
                });
            }

            // return all results ordered
            return results.Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).ToArray();
        }

        static double[][] CartesianProduct(GridParameter[] sequences)
        {
            var cartesian = CartesianProductEnumerable(sequences.Select(p => p.GetAllValues()));
            return cartesian.Select(row => row.ToArray()).ToArray();
        }

        static IEnumerable<IEnumerable<T>> CartesianProductEnumerable<T>(IEnumerable<IEnumerable<T>> sequences)
        {
            IEnumerable<IEnumerable<T>> emptyProduct = new[] { Enumerable.Empty<T>() };
            return sequences.Aggregate(
                emptyProduct,
                (accumulator, sequence) =>
                    from accseq in accumulator
                    from item in sequence
                    select accseq.Concat(new[] { item }));
        }
    }

}
