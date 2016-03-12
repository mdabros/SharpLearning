using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Random search optimizer initializes random parameters between min and max of the provided.
    /// Roughly based on: http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
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
        /// <param name="seed"></param>
        public RandomSearchOptimizer(double[][] parameterRanges, int iterations, int seed=42)
            : this(parameterRanges, iterations, seed, int.MaxValue)
        {
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parameterRanges">Each row is a series of values for a specific parameter</param>        
        /// <param name="iterations">The number of iterations to perform</param>
        /// <param name="seed"></param>
        /// <param name="maxDegreeOfParallelism">How many cores must be used for the optimization. 
        /// The function to minimize must be thread safe to use multi threading</param>
        public RandomSearchOptimizer(double[][] parameterRanges, int iterations, int seed, int maxDegreeOfParallelism)
        {
            if (parameterRanges == null) { throw new ArgumentNullException("parameterRanges"); }
            if (maxDegreeOfParallelism < 1) { throw new ArgumentException("maxDegreeOfParallelism must be at least 1"); }
            m_parameters = parameterRanges;
            m_maxDegreeOfParallelism = maxDegreeOfParallelism;
            m_random = new Random(seed);
            m_iterations = iterations;
        }

        /// <summary>
        /// Random search optimizer initializes random parameters between min and max of the provided.
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
        /// Random search optimizer initializes random parameters between min and max of the provided
        /// Returns all results ordered from best to worst (minimized).
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult[] Optimize(Func<double[], OptimizerResult> functionToMinimize)
        {
            // Generate the cartesian product between all parameters
            var grid = CreateNewSearchSpace(m_parameters);

            // Initialize the search
            var results = new ConcurrentBag<OptimizerResult>();
            var options = new ParallelOptions();
            options.MaxDegreeOfParallelism = m_maxDegreeOfParallelism;

            Parallel.ForEach(grid, options, param =>
            {
                // Get the current parameters for the current point
                var result = functionToMinimize(param);
                results.Add(result);
            });

            // return all results ordered
            return results.Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).ToArray();
        }


        double[][] CreateNewSearchSpace(double[][] parameters)
        {
            var newSearchSpace = new double[m_iterations][];
            for (int i = 0; i < newSearchSpace.Length; i++)
			{
                var newParameters = new double[parameters.Length];
                var index = 0;
                foreach (var param in parameters)
                {
                    newParameters[index] = NewParameter(param.Min(), param.Max());
                    index++;
                }
                newSearchSpace[i] = newParameters;
			}

            return newSearchSpace;
        }

        double NewParameter(double min, double max)
        {
            return m_random.NextDouble() * (max - min) + min;
        }
    }
}
