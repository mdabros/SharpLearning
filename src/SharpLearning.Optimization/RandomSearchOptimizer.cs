using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Random search optimizer initializes random parameters between min and max of the provided parameters.
    /// Roughly based on: http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
    /// </summary>
    public sealed class RandomSearchOptimizer : IOptimizer
    {
        readonly bool m_runParallel;
        readonly ParameterBounds[] m_parameters;
        readonly int m_iterations;
        readonly IParameterSampler m_sampler;

        /// <summary>
        /// Random search optimizer initializes random parameters between min and max of the provided parameters.
        /// Roughly based on: http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
        /// </summary>
        /// <param name="parameterRanges">A list of parameter bounds for each optimization parameter</param>        
        /// <param name="iterations">The number of iterations to perform</param>
        /// <param name="seed"></param>
        /// <param name="runParallel">Use multi threading to speed up execution (default is true)</param>
        public RandomSearchOptimizer(ParameterBounds[] parameterRanges, int iterations, int seed=42, bool runParallel = true)
        {
            if (parameterRanges == null) { throw new ArgumentNullException("parameterRanges"); }
            m_parameters = parameterRanges;
            m_runParallel = runParallel;
            m_sampler = new RandomUniform(seed);
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

            if(!m_runParallel)
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


        double[][] CreateNewSearchSpace(ParameterBounds[] parameters)
        {
            var newSearchSpace = new double[m_iterations][];
            for (int i = 0; i < newSearchSpace.Length; i++)
			{
                var newParameters = new double[parameters.Length];
                var index = 0;
                foreach (var param in parameters)
                {
                    newParameters[index] = param.NextValue(m_sampler);
                    index++;
                }
                newSearchSpace[i] = newParameters;
			}

            return newSearchSpace;
        }
    }
}
