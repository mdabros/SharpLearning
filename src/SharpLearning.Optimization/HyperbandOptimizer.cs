using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Optimization.ParameterSamplers;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Models;
using SharpLearning.Containers.Extensions;
using System.Diagnostics;

namespace SharpLearning.Optimization
{
    public delegate OptimizerResult HyperbandObjectiveFunction(double[] parameterSet, double unitsOfCompute);

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
        readonly int m_seed;

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
            if(maximumUnitsOfCompute < 1) throw new ArgumentException(nameof(maximumUnitsOfCompute) + " must be at larger than 0");
            if (eta < 1) throw new ArgumentException(nameof(eta) + " must be at larger than 0");
            m_sampler = new RandomUniform(seed);
            m_seed = seed;

            // This is called R in the paper.
            m_maximumUnitsOfCompute = maximumUnitsOfCompute;
            m_eta = eta;

            // This is called `s max` in the paper.
            m_numberOfRounds =  (int)(Math.Log(m_maximumUnitsOfCompute) / Math.Log(m_eta));
            // This is called `B` in the paper.
            m_totalUnitsOfComputePerRound = (m_numberOfRounds + 1) * m_maximumUnitsOfCompute;

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
            var bayesianParameterSampler = new BayesianConfigurationSampler(m_parameters, seed: m_seed);
            var randomConfigurationSampler = new RandomConfigurationSampler(m_parameters, m_seed);
            var randomInitialization = true;

            for (int rounds = m_numberOfRounds; rounds >= 0; rounds--)
            {
                // Initial configurations count.
                var initialConfigurationCount = (int)Math.Ceiling((m_totalUnitsOfComputePerRound / m_maximumUnitsOfCompute) 
                    * (Math.Pow(m_eta, rounds) / (rounds + 1)));

                // Initial unitsOfCompute per parameter set.
                var initialUnitsOfCompute = m_maximumUnitsOfCompute * Math.Pow(m_eta, -rounds);

                var randomParameterSetsCount = m_parameters.Length + 1;
                var parameterSets = new double[randomParameterSetsCount][];
                if(randomInitialization)
                {
                    for (int i = 0; i < randomParameterSetsCount; i++)
                    {
                        parameterSets[i] = randomConfigurationSampler.SampleConfiguration(allResults);
                    }                    
                }

                var results = new ConcurrentBag<OptimizerResult>();
                var iterations = m_skipLastIterationOfEachRound ? rounds : (rounds + 1);
                var firstIteration = true;
                for (int iteration = 0; iteration < iterations; iteration++)
                {
                    // Run each of the parameter sets with unitsOfCompute
                    // and keep the best (configurationCount / m_eta) configurations

                    var configurationCount = initialConfigurationCount * Math.Pow(m_eta, -iteration);
                    var unitsOfCompute = initialUnitsOfCompute * Math.Pow(m_eta, iteration);

                    Trace.WriteLine($"{(int)Math.Round(configurationCount)} configurations x {unitsOfCompute:F1} unitsOfCompute each");

                    if (firstIteration)
                    {
                        for (int i = 0; i < configurationCount; i++)
                        {
                            double[] parameterSet = null;
                            if (randomInitialization && i < parameterSets.Length)
                            {
                                parameterSet = parameterSets[i];
                            }
                            else
                            {
                                parameterSet = bayesianParameterSampler.SampleConfiguration(allResults);
                                randomInitialization = false;
                            }

                            var result = functionToMinimize(parameterSet, unitsOfCompute);
                            results.Add(result);
                            allResults.Add(result);
                        }
                        firstIteration = false;
                    }
                    else
                    {
                        foreach (var parameterSet in parameterSets)
                        {
                            var result = functionToMinimize(parameterSet, unitsOfCompute);
                            results.Add(result);
                            allResults.Add(result);
                        }
                    }

                    // Select a number of best configurations for the next loop
                    var configurationsToKeep = (int)Math.Round(configurationCount / m_eta);
                    Trace.WriteLine($"configurationsToKeep: {configurationsToKeep}");

                    parameterSets = results.OrderBy(v => v.Error)
                        .Take(configurationsToKeep)
                        .Select(v => v.ParameterSet)
                        .ToArray();
                }

                Trace.WriteLine($" Lowest loss so far: {allResults.OrderBy(v => v.Error).First().Error:F4}");
            }

            return allResults.ToArray();
        }

        public interface IConfigurationSampler
        {
            double[] SampleConfiguration(List<OptimizerResult> currentResults);
        }

        class RandomConfigurationSampler : IConfigurationSampler
        {
            readonly IParameterSpec[] m_parameters;
            readonly RandomUniform m_sampler;

            public RandomConfigurationSampler(IParameterSpec[] parameters, int seed)
            {
                m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
                m_sampler = new RandomUniform(seed);
            }

            public double[] SampleConfiguration(List<OptimizerResult> currentResults)
            {
                var newParameters = new double[m_parameters.Length];
                var index = 0;
                foreach (var param in m_parameters)
                {
                    newParameters[index] = param.SampleValue(m_sampler);
                    index++;
                }

                return newParameters;
            }
        }

        class BayesianConfigurationSampler : IConfigurationSampler
        {
            readonly IParameterSpec[] m_parameters;
            readonly IParameterSampler m_sampler;
            readonly Random m_random;

            // Important to use extra trees learner to have split between features calculated as: 
            // m_random.NextDouble() * (max - min) + min; 
            // instead of: (currentValue + prevValue) * 0.5; like in random forest.
            readonly RegressionExtremelyRandomizedTreesLearner m_learner;

            // Optimizer for finding maximum expectation (most promising hyper parameters) from extra trees model.
            readonly IOptimizer m_maximizer;

            // Acquisition function to maximize
            readonly AcquisitionFunction m_acquisitionFunc;

            public BayesianConfigurationSampler(IParameterSpec[] parameters, int seed)
            {
                m_parameters = parameters;
                m_random = new Random(seed);

                // Use member to seed the random uniform sampler.
                m_sampler = new RandomUniform(m_random.Next());

                // Hyper parameters for regression extra trees learner. These are based on the values suggested in http://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf.
                // However, according to the author Frank Hutter, the hyper parameters for the forest model should not matter that much.
                m_learner = new RegressionExtremelyRandomizedTreesLearner(trees: 30,
                    minimumSplitSize: 10,
                    maximumTreeDepth: 2000,
                    featuresPrSplit: parameters.Length,
                    minimumInformationGain: 1e-6,
                    subSampleRatio: 1.0,
                    seed: m_random.Next(), // Use member to seed the random uniform sampler.
                    runParallel: false);

                // Optimizer for finding maximum expectation (most promising hyper parameters) from extra trees model.
                m_maximizer = new RandomSearchOptimizer(m_parameters, iterations: 1000,
                    seed: m_random.Next(), // Use member to seed the random uniform sampler.
                    runParallel: false);

                // Acquisition function to maximize.
                m_acquisitionFunc = AcquisitionFunctions.ExpectedImprovement;
            }

            public double[] SampleConfiguration(List<OptimizerResult> currentResults)
            {
                // Fit model to current results
                var observations = currentResults.Select(r => r.ParameterSet)
                    .ToList().ToF64Matrix();
                var targets = currentResults.Select(r => r.Error)
                    .ToArray();
                var model = m_learner.Learn(observations, targets);

                var bestScore = targets.Min();

                // Find the most promising hyperparameters.
                Func<double[], OptimizerResult> minimize = (param) =>
                {
                    // use the model to predict the expected performance, mean and variance, of the parameter set.
                    var p = model.PredictCertainty(param);

                    return new OptimizerResult(param,
                        // negative, since we want to "maximize" the acquisition function.
                        -m_acquisitionFunc(bestScore, p.Prediction, p.Variance));
                };

                return m_maximizer.OptimizeBest(minimize).ParameterSet;
            }
        }
    }
}
