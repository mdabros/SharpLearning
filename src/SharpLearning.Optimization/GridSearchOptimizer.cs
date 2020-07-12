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
        readonly IParameterSpec[] m_parameters;
        readonly ParallelOptions m_parallelOptions;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parameters">A list of parameter specs, one for each optimization parameter</param>
        /// <param name="runParallel">Use multi threading to speed up execution (default is true)</param>
        /// <param name="maxDegreeOfParallelism">Maximum number of concurrent operations (default is -1 (unlimited))</param>
        public GridSearchOptimizer(IParameterSpec[] parameters, 
            bool runParallel = true, 
            int maxDegreeOfParallelism = -1)
        {
            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            m_runParallel = runParallel;
            m_parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDegreeOfParallelism
            };
        }

        /// <summary>
        /// Simple grid search that tries all combinations of the provided parameters.
        /// Returns the result which best minimizes the provided function.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult OptimizeBest(Func<double[], OptimizerResult> functionToMinimize) =>
            // Return the best model found.
            Optimize(functionToMinimize).Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).First();

        /// <summary>
        /// Simple grid search that tries all combinations of the provided parameters.
        /// Returns all results, chronologically ordered. 
        /// Note that the order of results might be affected if running parallel.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult[] Optimize(Func<double[], OptimizerResult> functionToMinimize)
        {
            // Generate the cartesian product between all parameters
            var parameterSets = CartesianProduct(m_parameters);

            var parameterIndexToResult = new ConcurrentDictionary<int, OptimizerResult>();
            if (!m_runParallel)
            {
                for (int index = 0; index < parameterSets.Length; index++)
                {
                    RunParameterSet(index, parameterSets,
                        functionToMinimize, parameterIndexToResult);
                }
            }
            else
            {
                Parallel.For(0, parameterSets.Length, m_parallelOptions, (index, loopState) =>
                {
                    RunParameterSet(index, parameterSets,
                        functionToMinimize, parameterIndexToResult);
                });
            }

            var results = parameterIndexToResult.OrderBy(v => v.Key)
                .Select(v => v.Value)
                .ToArray();

            return results;
        }

        static double[][] CartesianProduct(IParameterSpec[] sequences)
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

        static void RunParameterSet(int paramterSetIndex,
            double[][] parameterSets,
            Func<double[], OptimizerResult> functionToMinimize,
            ConcurrentDictionary<int, OptimizerResult> parameterIndexToResult)
        {
            var parameterSet = parameterSets[paramterSetIndex];
            var result = functionToMinimize(parameterSet);
            parameterIndexToResult.AddOrUpdate(paramterSetIndex, result, (index, r) => r);
        }
    }
}
