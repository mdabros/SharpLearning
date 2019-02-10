using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Optimization.ParameterSamplers;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Models;
using SharpLearning.Containers.Extensions;
using System.Diagnostics;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Optimization
{
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
    public sealed class BOHPOptimizer
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
        public BOHPOptimizer(IParameterSpec[] parameters, 
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
            var configurationSampler = new BOHPConfigurationSampler(m_parameters, 342, 0.15, m_parameters.Length + 1, 0.2);

            for (int rounds = m_numberOfRounds; rounds >= 0; rounds--)
            {
                // Initial configurations count.
                var initialConfigurationCount = (int)Math.Ceiling((m_totalUnitsOfComputePerRound / m_maximumUnitsOfCompute) 
                    * (Math.Pow(m_eta, rounds) / (rounds + 1)));

                // Initial unitsOfCompute per parameter set.
                var initialUnitsOfCompute = m_maximumUnitsOfCompute * Math.Pow(m_eta, -rounds);
                var iterations = m_skipLastIterationOfEachRound ? rounds : (rounds + 1);

                for (int iteration = 0; iteration < iterations; iteration++)
                {
                    // Run each of the parameter sets with unitsOfCompute
                    // and keep the best (configurationCount / m_eta) configurations
                    var configurationCount = initialConfigurationCount * Math.Pow(m_eta, -iteration);
                    var unitsOfCompute = initialUnitsOfCompute * Math.Pow(m_eta, iteration);

                    Trace.WriteLine($"{(int)Math.Round(configurationCount)} configurations x {unitsOfCompute:F1} unitsOfCompute each");

                    OptimizerResult result = null;
                    for (int i = 0; i < configurationCount; i++)
                    {
                        var budget = (int)Math.Round(unitsOfCompute);
                        var parameterSet = configurationSampler.Sample(result, budget);
                        result = functionToMinimize(parameterSet, unitsOfCompute);

                        allResults.Add(result);
                    }

                    Trace.WriteLine($" Lowest loss so far: {allResults.OrderBy(v => v.Error).First().Error:F4}");
                }
            }

            return allResults.ToArray();
        }

        public interface IConfigurationSampler
        {
            double[] Sample(OptimizerResult newResult);
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

            public double[] Sample(OptimizerResult newResult)
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
               
        class BayesianConfigurationSampler
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

            public double[] Sample(List<OptimizerResult> currentResults)
            {
                // Fit model to current results
                var observations = currentResults.Select(r => r.ParameterSet)
                    .ToList().ToF64Matrix();
                var targets = currentResults.Select(r => r.Error)
                    .ToArray();
                var model = m_learner.Learn(observations, targets);

                var bestScore = targets.Min();

                // Find the most promising hyperparameters.
                var minimize = Minimize(model, bestScore);

                return m_maximizer.OptimizeBest(minimize).ParameterSet;
            }

            Func<double[], OptimizerResult> Minimize(RegressionForestModel model, double bestScore)
            {
                Func<double[], OptimizerResult> minimize = (param) =>
                {
                    // use the model to predict the expected performance, mean and variance, of the parameter set.
                    var p = model.PredictCertainty(param);

                    return new OptimizerResult(param,
                        // negative, since we want to "maximize" the acquisition function.
                        -m_acquisitionFunc(bestScore, p.Prediction, p.Variance));
                };
                return minimize;
            }
        }

        class BOHPConfigurationSampler
        {
            readonly IParameterSpec[] m_parameters;
            enum SamplerModel { Good, Bad };

            Dictionary<int, Dictionary<SamplerModel, RegressionForestModel>> m_models;
            readonly RegressionExtremelyRandomizedTreesLearner m_learner;
            readonly RandomConfigurationSampler m_randomConfigurationSampler;

            // Optimizer for finding maximum expectation (most promising hyper parameters) from extra trees model.
            readonly IOptimizer m_maximizer;

            // Acquisition function to maximize
            readonly AcquisitionFunction m_acquisitionFunc;

            readonly double m_topSampleRatioToTrainOn;
            readonly int m_minimiumTrainingSamples;
            readonly double m_randomConfigurationRatio;

            readonly Random m_random;

            Dictionary<int, List<OptimizerResult>> m_budgetToResults;

            public BOHPConfigurationSampler(IParameterSpec[] parameters, int seed,
                double topSampleRatioToTrainOn, int minimiumTrainingSamples, double randomConfigurationRatio)
            {
                m_parameters = parameters;
                m_random = new Random(seed);

                m_randomConfigurationSampler = new RandomConfigurationSampler(parameters, m_random.Next());

                m_topSampleRatioToTrainOn = topSampleRatioToTrainOn;
                m_minimiumTrainingSamples = minimiumTrainingSamples;
                m_randomConfigurationRatio = randomConfigurationRatio;

                m_models = new Dictionary<int, Dictionary<SamplerModel, RegressionForestModel>>();
                m_budgetToResults = new Dictionary<int, List<OptimizerResult>>();

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

            public double[] Sample(OptimizerResult newResult, int budget)
            {
                if(newResult == null || m_random.NextDouble() < m_randomConfigurationRatio)
                {
                    return m_randomConfigurationSampler.Sample(newResult);
                }
                else
                {
                    AddResultsToBudget(newResult, budget);

                    // Check if any budget contains enough samples for a model.
                    if(!m_budgetToResults.Where(v => v.Value.Count > m_minimiumTrainingSamples).Any())
                    {
                        return m_randomConfigurationSampler.Sample(newResult);
                    }

                    // Sample from the largest budget model available.
                    var maxBudget = m_budgetToResults
                        .Where(v => v.Value.Count > m_minimiumTrainingSamples)
                        .Max(v => v.Key);

                    var results = m_budgetToResults[maxBudget];

                    TrainBudgetModels(results, maxBudget);

                    // Always sample from max budget models if available
                    var good = m_models[maxBudget][SamplerModel.Good];
                    var bad = m_models[maxBudget][SamplerModel.Bad];

                    var best = results.OrderBy(r => r.Error).First().Error;

                    var l = Minimize(good, best);
                    var g = Minimize(bad, best);

                    Func<double[], OptimizerResult> minimize = p =>
                    {
                        var gr = -g(p).Error;
                        var lr = -l(p).Error;
                        // negative since we want to maximize using the minimizer.
                        var r = -Math.Max(1e-32, gr) / Math.Max(1e-32, lr);
                        return new OptimizerResult(p, r);
                    };

                    return m_maximizer.OptimizeBest(minimize).ParameterSet;
                }
            }

            void AddResultsToBudget(OptimizerResult result, int budget)
            {
                if (!m_budgetToResults.ContainsKey(budget))
                {
                    m_budgetToResults.Add(budget, new List<OptimizerResult> { result });
                }
                else
                {
                    m_budgetToResults[budget].Add(result);
                }
            }

            void TrainBudgetModels(List<OptimizerResult> results, int budget)
            {
                var result = m_budgetToResults[budget];

                var resultCount = result.Count;

                var goodResultCount = Math.Max(m_minimiumTrainingSamples, 
                    (int)Math.Round(m_topSampleRatioToTrainOn * resultCount));
                var badResultCount = Math.Max(m_minimiumTrainingSamples, 
                    resultCount - goodResultCount);

                // take best results
                var goodResults = result.OrderBy(v => v.Error)
                    .Take(goodResultCount);

                // take worst results
                var badResults = result.OrderByDescending(v => v.Error)
                    .Take(badResultCount);

                var goodModel = TrainModel(goodResults);
                var badModel = TrainModel(badResults);

                var budgetModels = new Dictionary<SamplerModel, RegressionForestModel>
                {
                    { SamplerModel.Good, goodModel },
                    { SamplerModel.Bad, badModel},
                };

                if (m_models.ContainsKey(budget))
                {
                    m_models[budget] = budgetModels;
                }
                else
                {
                    m_models.Add(budget, budgetModels);
                }
            }

            Func<double[], OptimizerResult> Minimize(RegressionForestModel model, double bestScore)
            {
                Func<double[], OptimizerResult> minimize = (param) =>
                {
                    // use the model to predict the expected performance, mean and variance, of the parameter set.
                    var p = model.PredictCertainty(param);

                    // negative, since we want to "maximize" the acquisition function.
                    var result = -m_acquisitionFunc(bestScore, p.Prediction, p.Variance);

                    return new OptimizerResult(param, result);
                };
                return minimize;
            }

            RegressionForestModel TrainModel(IEnumerable<OptimizerResult> results)
            {
                var trainingData = ConvertToTrainingData(results);
                var model = m_learner.Learn(trainingData.observations, trainingData.targets);

                return model;
            }

            (F64Matrix observations, double[] targets) ConvertToTrainingData(IEnumerable<OptimizerResult> results)
            {
                var observations = results.Select(r => r.ParameterSet)
                    .ToList().ToF64Matrix();
                var targets = results.Select(r => r.Error)
                    .ToArray();

                return (observations, targets);
            }
        }
    }
}
