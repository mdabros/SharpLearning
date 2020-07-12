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
        readonly IParameterSpec[] m_parameters;
        readonly int m_iterations;
        readonly IParameterSampler m_sampler;
        readonly ParallelOptions m_parallelOptions;

        /// <summary>
        /// Random search optimizer initializes random parameters between min and max of the provided parameters.
        /// Roughly based on: http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
        /// </summary>
        /// <param name="parameters">A list of parameter specs, one for each optimization parameter</param>        
        /// <param name="iterations">The number of iterations to perform</param>
        /// <param name="seed"></param>
        /// <param name="runParallel">Use multi threading to speed up execution (default is true)</param>
        /// <param name="maxDegreeOfParallelism">Maximum number of concurrent operations (default is -1 (unlimited))</param>
        public RandomSearchOptimizer(IParameterSpec[] parameters,
            int iterations,
            int seed = 42,
            bool runParallel = true,
            int maxDegreeOfParallelism = -1)
        {
            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            m_runParallel = runParallel;
            m_sampler = new RandomUniform(seed);
            m_iterations = iterations;
            m_parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDegreeOfParallelism
            };
        }

        /// <summary>
        /// Random search optimizer initializes random parameters between min and max of the provided bounds.
        /// Returns the result which best minimizes the provided function.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult OptimizeBest(Func<double[], OptimizerResult> functionToMinimize) =>
            // Return the best model found.
            Optimize(functionToMinimize).Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).First();

        /// <summary>
        /// Random search optimizer initializes random parameters between min and max of the provided bounds.
        /// Returns all results, chronologically ordered. 
        /// Note that the order of results might be affected if running parallel.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult[] Optimize(Func<double[], OptimizerResult> functionToMinimize)
        {
            // Generate parameter sets.
            var parameterSets = SampleRandomParameterSets(m_iterations, 
                m_parameters, m_sampler);

            // Run parameter sets.
            var parameterIndexToResult = new ConcurrentDictionary<int, OptimizerResult>();
            if(!m_runParallel)
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

        /// <summary>
        /// Samples a set of random parameter sets.
        /// </summary>
        /// <param name="parameterSetCount"></param>
        /// <param name="parameters"></param>
        /// <param name="sampler"></param>
        /// <returns></returns>
        public static double[][] SampleRandomParameterSets(int parameterSetCount, 
            IParameterSpec[] parameters, IParameterSampler sampler)
        {
            var parameterSets = new double[parameterSetCount][];
            for (int i = 0; i < parameterSetCount; i++)
            {
                parameterSets[i] = SampleParameterSet(parameters, sampler);
            }

            return parameterSets;
        }

        /// <summary>
        /// Samples a random parameter set.
        /// </summary>
        /// <param name="parameters"></param>
        /// <param name="sampler"></param>
        /// <returns></returns>
        public static double[] SampleParameterSet(IParameterSpec[] parameters, 
            IParameterSampler sampler)
        {
            var parameterSet = new double[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                var parameter = parameters[i];
                parameterSet[i] = parameter.SampleValue(sampler);
            }

            return parameterSet;
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
