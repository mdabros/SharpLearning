using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization
{
    public delegate OptimizerResult HyperbandObjectiveFunction(double[] parameterSet, double unitsOfCompute);

    /// <summary>
    /// Hyperband optimizer based on:
    /// https://arxiv.org/pdf/1603.06560.pdf
    /// 
    /// Implementation based on:
    /// https://github.com/zygmuntz/hyperband
    /// 
    /// </summary>
    public sealed class HyperbandOptimizer
    {
        readonly IParameterSpec[] m_parameters;
        readonly IParameterSampler m_sampler;

        readonly int m_maximumUnitsOfCompute;
        readonly int m_eta;

        readonly int m_numberOfRounds;
        readonly int m_totalUnitsOfComputePerRound;

        readonly bool m_skipLastIterationOfEachRound;

        /// <summary>
        /// Hyperband optimizer based on: https://arxiv.org/pdf/1603.06560.pdf
        /// 
        /// Hyperband controls a budget of compute for each set of hyperparameters, 
        /// Initially it will run each parameter set with very little compute budget to get a taste of how they perform. 
        /// Then it takes the best performers and runs them on a larger budget. 
        /// </summary>
        /// <param name="parameters">A list of parameter specs, one for each optimization parameter</param>
        /// <param name="maximumUnitsOfCompute">This indicates the maximum units of compute.
        /// A unit of compute could be 5 epochs over a dataset for instance. Consequently, 
        /// a unit of compute should be chosen to be the minimum amount of computation where different 
        /// hyperparameter configurations start to separate (or where it is clear that some settings diverge)></param>
        /// <param name="eta">Controls the proportion of configurations discarded in each round.
        /// Together with maximumUnitsOfCompute, it dictates how many rounds are considered</param>
        /// <param name="skipLastIterationOfEachRound">True to skip the last, 
        /// most computationally expensive, iteration of each round. Default is false.</param>
        public HyperbandOptimizer(IParameterSpec[] parameters, 
            int maximumUnitsOfCompute = 81, int eta = 3,
            bool skipLastIterationOfEachRound = false,
            int seed = 34)
        {
            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            m_sampler = new RandomUniform(seed);

            // This is called R in the paper.
            m_maximumUnitsOfCompute = maximumUnitsOfCompute;
            m_eta = eta;

            // This is called `s max` in the paper.
            m_numberOfRounds =  (int)(Math.Log(m_maximumUnitsOfCompute) / Math.Log(m_eta));
            // This is called `B` in the paper.
            m_totalUnitsOfComputePerRound = (m_numberOfRounds + 1) * m_maximumUnitsOfCompute;

            m_skipLastIterationOfEachRound = skipLastIterationOfEachRound;
        }

        public OptimizerResult[] Optimize(HyperbandObjectiveFunction functionToMinimize)
        {
            var allResults = new List<OptimizerResult>();

            for (int rounds = m_numberOfRounds; rounds >= 0; rounds--)
            {
                // Initial configurations count.
                var initialConfigurationCount = (int)Math.Ceiling((m_totalUnitsOfComputePerRound / m_maximumUnitsOfCompute) 
                    * (Math.Pow(m_eta, rounds) / (rounds + 1)));

                // Initial unitsOfCompute per parameter set.
                var initialUnitsOfCompute = m_maximumUnitsOfCompute * Math.Pow(m_eta, -rounds);

                var parameterSets = CreateParameterSets(m_parameters, initialConfigurationCount);
                var results = new ConcurrentBag<OptimizerResult>();

                var iterations = m_skipLastIterationOfEachRound ? rounds : (rounds + 1);
                for (int iteration = 0; iteration < iterations; iteration++)
                {
                    // Run each of the parameter sets with unitsOfCompute
                    // and keep the best (configurationCount / m_eta) configurations

                    var configurationCount = initialConfigurationCount * Math.Pow(m_eta, -iteration);
                    var unitsOfCompute = initialUnitsOfCompute * Math.Pow(m_eta, iteration);

                    //Trace.WriteLine($"{(int)Math.Round(configurationCount)} configurations x {unitsOfCompute:F1} unitsOfCompute each");
                    foreach (var parameterSet in parameterSets)
                    {
                        var result = functionToMinimize(parameterSet, unitsOfCompute);
                        results.Add(result);
                    }

                    // Select a number of best configurations for the next loop
                    var configurationsToKeep = (int)Math.Round(configurationCount / m_eta);
                    parameterSets = results.OrderBy(v => v.Error)
                        .Take(configurationsToKeep)
                        .Select(v => v.ParameterSet)
                        .ToArray();
                }

                allResults.AddRange(results);
                //Trace.WriteLine($" Lowest loss so far: {allResults.OrderBy(v => v.Error).First().Error:F4}");
            }

            return allResults.ToArray();
        }

        double[][] CreateParameterSets(IParameterSpec[] parameters, 
            int setCount)
        {
            var newSearchSpace = new double[setCount][];
            for (int i = 0; i < newSearchSpace.Length; i++)
            {
                var newParameters = new double[parameters.Length];
                var index = 0;
                foreach (var param in parameters)
                {
                    newParameters[index] = param.SampleValue(m_sampler);
                    index++;
                }
                newSearchSpace[i] = newParameters;
            }

            return newSearchSpace;
        }
    }
}
