using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
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
        readonly Dictionary<string, IParameterSpec> m_parameters;
        readonly int m_iterations;
        readonly IParameterSampler m_sampler;

        /// <summary>
        /// Random search optimizer initializes random parameters between min and max of the provided parameters.
        /// Roughly based on: http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
        /// </summary>
        /// <param name="parameters">A list of parameter specs, one for each optimization parameter</param>        
        /// <param name="iterations">The number of iterations to perform</param>
        /// <param name="seed"></param>
        /// <param name="runParallel">Use multi threading to speed up execution (default is true)</param>
        public RandomSearchOptimizer(Dictionary<string, IParameterSpec> parameters, int iterations, int seed=42, bool runParallel = true)
        {
            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            m_runParallel = runParallel;
            m_sampler = new RandomUniform(seed);
            m_iterations = iterations;
        }

        /// <summary>
        /// Random search optimizer initializes random parameters between min and max of the provided.
        /// Returns the result which best minimizes the provided function.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult OptimizeBest(FunctionToMinimize functionToMinimize) =>
            // Return the best model found.
            Optimize(functionToMinimize).First();

        /// <summary>
        /// Random search optimizer initializes random parameters between min and max of the provided
        /// Returns all results ordered from best to worst (minimized).
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult[] Optimize(FunctionToMinimize functionToMinimize)
        {
            // Generate the cartesian product between all parameters
            var parameterSets = CreateParameterSets(m_parameters);

            // Initialize the search
            var results = new ConcurrentBag<OptimizerResult>();

            if(!m_runParallel)
            {
                foreach (var parameterSet in parameterSets)
                {
                    // Get the current parameters for the current point
                    var result = functionToMinimize(parameterSet);
                    results.Add(result);
                }
            }
            else
            {
                var rangePartitioner = Partitioner.Create(parameterSets, true);
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


        List<Dictionary<string, double>> CreateParameterSets(Dictionary<string, IParameterSpec> parameters)
        {
            var searchSpace = new List<Dictionary<string, double>>();
            for (int i = 0; i < m_iterations; i++)
			{
                var parameterSet = new Dictionary<string, double>();

                // Order by name to ensure reproducibility.
                foreach (var nameToParameter in m_parameters.OrderBy(v => v.Key))
                {
                    var name = nameToParameter.Key;
                    var parameterSpec = nameToParameter.Value;

                    parameterSet.Add(nameToParameter.Key, parameterSpec.SampleValue(m_sampler));
                }

                searchSpace.Add(parameterSet);
			}

            return searchSpace;
        }
    }
}
