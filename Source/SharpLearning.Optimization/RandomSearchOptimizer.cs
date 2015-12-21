using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Random search optimizer initializes random parameters between min and max of the provided
    /// parameters.
    /// </summary>
    public sealed class RandomSearchOptimizer : IOptimizer
    {
        readonly int m_maxDegreeOfParallelism;
        readonly double[][] m_parameters;
        readonly int m_iterations;
        readonly Random m_random;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parameterRanges">Each row is a series of values for a specific parameter</param>
        /// <param name="iterations">The number of iterations to perform</param>
        public RandomSearchOptimizer(double[][] parameterRanges, int iterations)
            : this(parameterRanges, iterations, int.MaxValue)
        {
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parameterRanges">Each row is a series of values for a specific parameter</param>
        /// <param name="iterations">The number of iterations to perform</param>
        /// <param name="maxDegreeOfParallelism">How many cores must be used for the optimization. 
        /// The function to minimize must be thread safe to use multi threading</param>
        public RandomSearchOptimizer(double[][] parameterRanges, int iterations, int maxDegreeOfParallelism)
        {
            if (parameterRanges == null) { throw new ArgumentNullException("parameterRanges"); }
            if (maxDegreeOfParallelism < 1) { throw new ArgumentException("maxDegreeOfParallelism must be at least 1"); }
            m_parameters = parameterRanges;
            m_maxDegreeOfParallelism = maxDegreeOfParallelism;
            m_random = new Random(42);
            m_iterations = iterations;
        }

        /// <summary>
        /// Random search optimizer initializes random parameters between min and max of the provided
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult Optimize(Func<double[], OptimizerResult> functionToMinimize)
        {
            // Generate the cartesian product between all parameters
            double[][] grid = CartesianProduct(CreateNewSearchSpace(m_parameters));

            // Initialize the search
            var results = new ConcurrentBag<OptimizerResult>();
            var options = new ParallelOptions();
            options.MaxDegreeOfParallelism = m_maxDegreeOfParallelism;

            Parallel.ForEach(grid, options, param =>
            {
                // Get the current parameters for the current point
                var result = functionToMinimize(param);
                //Trace.WriteLine("Error: " + result.Error);//
                results.Add(result);
            });

            // Return the best model found.
            return results.Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).First();
        }

        double[][] CreateNewSearchSpace(double[][] parameters)
        {
            var newSearchSpace = new double[parameters.Length][];
            var parameterCount = m_iterations / parameters.Length;
            for (int i = 0; i < parameters.Length; i++)
			{
                var inputParams = parameters[i];
                newSearchSpace[i] = Boundaries(inputParams.Min(), inputParams.Max(), parameterCount);
			}

            return newSearchSpace;
        }

        double[] Boundaries(double min, double max, int parameterCounts)
        {
            var parameters = new double[parameterCounts];
            for (int i = 0; i < parameterCounts; i++)
            {
                parameters[i] = m_random.NextDouble() * (max - min) + min;
            }

            return parameters;
        }

        static T[][] CartesianProduct<T>(T[][] sequences)
        {
            var cartesian = CartesianProductEnumerable(sequences);
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
