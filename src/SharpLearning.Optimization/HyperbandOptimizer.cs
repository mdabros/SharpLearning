using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Objective function for hyperband optimizer.
    /// </summary>
    /// <param name="parameterSet">Parameter set to run.</param>
    /// <param name="budget">Budget under which to run the parameter set.</param>
    /// <returns></returns>
    public delegate OptimizerResult HyperbandObjectiveFunction(double[] parameterSet, double budget);

    /// <summary>
    /// Hyperband optimizer based on:
    /// https://arxiv.org/pdf/1603.06560.pdf
    /// Implementation based on:
    /// https://github.com/zygmuntz/hyperband
    /// 
    /// Hyperband controls a budget of compute for each set of hyperparameters, 
    /// Initially it will run each parameter set with very little compute budget to get a taste of how they perform. 
    /// Then it takes the best performers and runs them on a larger budget. 
    /// </summary>
    public sealed class HyperbandOptimizer
    {
        readonly IParameterSpec[] m_parameters;
        readonly IParameterSampler m_sampler;

        readonly int m_maximumBudget;
        readonly int m_eta;

        readonly int m_numberOfRounds;
        readonly int m_totalBudgetPerRound;

        readonly bool m_skipLastIterationOfEachRound;

        /// <summary>
        /// Hyperband optimizer based on: https://arxiv.org/pdf/1603.06560.pdf
        /// 
        /// Hyperband controls a budget of compute for each set of hyperparameters, 
        /// Initially it will run each parameter set with very little compute budget to get a taste of how they perform. 
        /// Then it takes the best performers and runs them on a larger budget. 
        /// </summary>
        /// <param name="parameters">A list of parameter specs, one for each optimization parameter</param>
        /// <param name="maximumBudget">This provides the maximum budget.
        /// One unit of compute could be 5 epochs over a dataset for instance. Consequently, 
        /// a unit of compute should be chosen to be the minimum amount of computation where different 
        /// hyperparameter configurations start to separate (or where it is clear that some settings diverge)></param>
        /// <param name="eta">Controls the proportion of configurations discarded in each round.
        /// Together with maximumUnitsOfCompute, it dictates how many rounds are considered</param>
        /// <param name="skipLastIterationOfEachRound">True to skip the last, 
        /// most computationally expensive, iteration of each round. Default is false.</param>
        /// <param name="seed"></param>
        public HyperbandOptimizer(IParameterSpec[] parameters, 
            int maximumBudget = 81, int eta = 3,
            bool skipLastIterationOfEachRound = false,
            int seed = 34)
        {
            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            if(maximumBudget < 1) throw new ArgumentException(nameof(maximumBudget) + " must be at larger than 0");
            if (eta < 1) throw new ArgumentException(nameof(eta) + " must be at larger than 0");
            m_sampler = new RandomUniform(seed);

            // This is called R in the paper.
            m_maximumBudget = maximumBudget;
            m_eta = eta;

            // This is called `s max` in the paper.
            m_numberOfRounds =  (int)(Math.Log(m_maximumBudget) / Math.Log(m_eta));
            // This is called `B` in the paper.
            m_totalBudgetPerRound = (m_numberOfRounds + 1) * m_maximumBudget;

            // Suggestion by fastml: http://fastml.com/tuning-hyperparams-fast-with-hyperband/
            // "One could discard the last tier (1 x 81, 2 x 81, etc.) in each round, 
            // including the last round. This drastically reduces time needed.             
            m_skipLastIterationOfEachRound = skipLastIterationOfEachRound;
        }

        /// <summary>
        /// Optimization using Hyberband.
        /// Returns the result which best minimizes the provided function.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult OptimizeBest(HyperbandObjectiveFunction functionToMinimize) =>
            // Return the best model found.
            Optimize(functionToMinimize).Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).First();

        /// <summary>
        /// Optimization using Hyberband.
        /// Returns all results, chronologically ordered.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult[] Optimize(HyperbandObjectiveFunction functionToMinimize)
        {
            var allResults = new List<OptimizerResult>();

            for (int rounds = m_numberOfRounds; rounds >= 0; rounds--)
            {
                // Initial configurations count.
                var initialConfigurationCount = (int)Math.Ceiling((m_totalBudgetPerRound / m_maximumBudget) 
                    * (Math.Pow(m_eta, rounds) / (rounds + 1)));

                // Initial budget per parameter set.
                var initialBudget = m_maximumBudget * Math.Pow(m_eta, -rounds);

                var parameterSets = RandomSearchOptimizer.SampleRandomParameterSets(initialConfigurationCount,
                    m_parameters, m_sampler);

                var results = new ConcurrentBag<OptimizerResult>();

                var iterations = m_skipLastIterationOfEachRound ? rounds : (rounds + 1);
                for (int iteration = 0; iteration < iterations; iteration++)
                {
                    // Run each of the parameter sets with budget
                    // and keep the best (configurationCount / m_eta) configurations

                    var configurationCount = initialConfigurationCount * Math.Pow(m_eta, -iteration);
                    var budget = initialBudget * Math.Pow(m_eta, iteration);

                    //Trace.WriteLine($"{(int)Math.Round(configurationCount)} configurations x {budget:F1} budget each");
                    foreach (var parameterSet in parameterSets)
                    {
                        var result = functionToMinimize(parameterSet, budget);
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
    }
}
