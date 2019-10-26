using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.Optimization.ParameterSamplers;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Models;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Bayesian optimization (BO) for global black box optimization problems. BO learns a model based on the initial parameter sets and scores.
    /// This model is used to sample new promising parameter candidates which are evaluated and added to the existing parameter sets.
    /// This process iterates several times. The method is computational expensive so is most relevant for expensive problems, 
    /// where each evaluation of the function to minimize takes a long time, like hyper parameter tuning a machine learning method.
    /// But in that case it can usually reduce the number of iterations required to reach a good solution compared to less sophisticated methods.
    /// Implementation loosely based on:
    /// http://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf
    /// https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
    /// https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
    /// </summary>
    public sealed class BayesianOptimizer : IOptimizer
    {
        readonly Random m_random;
        readonly IParameterSpec[] m_parameters;
        readonly IParameterSampler m_sampler;
        readonly int m_iterations;
        readonly int m_randomStartingPointsCount;
        readonly int m_functionEvaluationsPerIterationCount;
        readonly int m_randomSearchPointCount;
        
        readonly int m_maxDegreeOfParallelism;
        readonly bool m_runParallel;

        // Important to use extra trees learner to have split between features calculated as: 
        // m_random.NextDouble() * (max - min) + min; 
        // instead of: (currentValue + prevValue) * 0.5; like in random forest.
        readonly RegressionExtremelyRandomizedTreesLearner m_learner;

        /// <summary>
        /// Bayesian optimization (BO) for global black box optimization problems. BO learns a model based on the initial parameter sets and scores.
        /// This model is used to sample new promising parameter candidates which are evaluated and added to the existing parameter sets.
        /// This process iterates several times. The method is computational expensive so is most relevant for expensive problems, 
        /// where each evaluation of the function to minimize takes a long time, like hyper parameter tuning a machine learning method.
        /// But in that case it can usually reduce the number of iterations required to reach a good solution compared to less sophisticated methods.
        /// Implementation loosely based on:
        /// http://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf
        /// https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
        /// https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
        /// </summary>
        /// <param name="parameters">A list of parameter specs, one for each optimization parameter</param>
        /// <param name="iterations">The number of iterations to perform.
        /// Iteration * functionEvaluationsPerIteration = totalFunctionEvaluations</param>
        /// <param name="randomStartingPointCount">Number of randomly parameter sets used 
        /// for initialization (default is 20)</param>
        /// <param name="functionEvaluationsPerIterationCount">The number of function evaluations per iteration. 
        /// The parameter sets are included in order of most promising outcome (default is 1)</param>
        /// <param name="randomSearchPointCount">The number of random parameter sets
        /// used when maximizing the expected improvement acquisition function (default is 1000)</param>
        /// <param name="seed"></param>
        /// <param name="runParallel">Use multi threading to speed up execution (default is false).
        /// Note that the order of results returned the Optimize method will not be reproducible when running in parallel.
        /// Results will be the same, only the order is not reproducible</param>
        /// <param name="maxDegreeOfParallelism">Maximum number of concurrent operations (default is -1 (unlimited))</param>
        public BayesianOptimizer(IParameterSpec[] parameters,
            int iterations,
            int randomStartingPointCount = 5,
            int functionEvaluationsPerIterationCount = 1,
            int randomSearchPointCount = 1000,
            int seed = 42,
            bool runParallel = false,
            int maxDegreeOfParallelism = -1)
        {
            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));

            if (iterations < 1) throw new ArgumentException(nameof(iterations) +
                 "must be at least 1. Was: " + iterations);
            if (randomStartingPointCount < 1) throw new ArgumentException(nameof(randomStartingPointCount) +
                "must be at least 1. Was: " + randomStartingPointCount);
            if (functionEvaluationsPerIterationCount < 1) throw new ArgumentException(nameof(functionEvaluationsPerIterationCount) +
                "must be at least 1. Was: " + functionEvaluationsPerIterationCount);
            if (randomSearchPointCount < 1) throw new ArgumentException(nameof(randomSearchPointCount) +
                "must be at least 1. Was: " + randomSearchPointCount);

            m_random = new Random(seed);
            // Use member to seed the random uniform sampler.
            m_sampler = new RandomUniform(m_random.Next());
            m_iterations = iterations;
            m_randomStartingPointsCount = randomStartingPointCount;
            m_functionEvaluationsPerIterationCount = functionEvaluationsPerIterationCount;
            m_randomSearchPointCount = randomSearchPointCount;

            // Hyper parameters for regression extra trees learner. 
            // These are based on the values suggested in http://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf.
            // However, according to the author Frank Hutter, 
            // the hyper parameters for the forest model should not matter that much.
            m_learner = new RegressionExtremelyRandomizedTreesLearner(trees: 30,
                minimumSplitSize: 10,
                maximumTreeDepth: 2000,
                featuresPrSplit: parameters.Length,
                minimumInformationGain: 1e-6,
                subSampleRatio: 1.0,
                seed: m_random.Next(), // Use member to seed the random uniform sampler.
                runParallel: false);

            m_runParallel = runParallel;
            m_maxDegreeOfParallelism = maxDegreeOfParallelism;
        }

        /// <summary>
        /// Optimization using Sequential Model-based optimization.
        /// Returns the result which best minimizes the provided function.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult OptimizeBest(Func<double[], OptimizerResult> functionToMinimize) =>
            // Return the best model found.
            Optimize(functionToMinimize).Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).First();

        /// <summary>
        /// Minimizes the provided function
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult[] Optimize(Func<double[], OptimizerResult> functionToMinimize)
        {
            var initialParameterSets = ProposeParameterSets(m_randomStartingPointsCount, null);

            // Initialize the search
            var results = new List<OptimizerResult>();
            var initializationResults = RunParameterSets(functionToMinimize, initialParameterSets);
            results.AddRange(initializationResults);

            for (int iteration = 0; iteration < m_iterations; iteration++)
            {
                var parameterSets = ProposeParameterSets(m_functionEvaluationsPerIterationCount, results);
                var iterationResults = RunParameterSets(functionToMinimize, parameterSets);
                results.AddRange(iterationResults);
            }

            return results.ToArray();
        }

        /// <summary>
        /// Runs a set of parameter sets and returns the results.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <param name="parameterSets"></param>
        /// <returns></returns>
        public List<OptimizerResult> RunParameterSets(Func<double[], OptimizerResult> functionToMinimize,
            double[][] parameterSets)
        {
            var results = new ConcurrentBag<OptimizerResult>();
            if (!m_runParallel)
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
                var options = new ParallelOptions { MaxDegreeOfParallelism = m_maxDegreeOfParallelism };
                Parallel.ForEach(rangePartitioner, options, (param, loopState) =>
                {
                    // Get the current parameters for the current point
                    var result = functionToMinimize(param);
                    results.Add(result);
                });
            }

            return results.ToList();
        }

        /// <summary>
        /// Propose a new list of parameter sets.
        /// </summary>
        /// <param name="parameterSetCount">The number of parameter sets to propose</param>
        /// <param name="previousResults">Results from previous runs.  
        /// These are used in the model for proposing new parameter sets.
        /// If no results are provided, random parameter sets will be returned.</param>
        /// <returns></returns>
        public double[][] ProposeParameterSets(int parameterSetCount,
            IReadOnlyList<OptimizerResult> previousResults = null)
        {
            var previousParameterSetCount = previousResults == null ? 0 : previousResults.Count;
            if (previousParameterSetCount < m_randomStartingPointsCount)
            {
                var randomParameterSetCount = Math.Min(parameterSetCount,
                    m_randomStartingPointsCount - previousParameterSetCount);

                var randomParameterSets = RandomSearchOptimizer.SampleRandomParameterSets(
                    randomParameterSetCount, m_parameters, m_sampler);

                return randomParameterSets;
            }

            // Filter away NaNs, and ensure result order is preserved, when fitting the model.
            var validParameterSets = previousResults
                .Where(v => !double.IsNaN(v.Error))
                .OrderBy(v => v.Error); // TODO: This might still fail to provide same order if two different parameter sets yield the same error.
            
            var model = FitModel(validParameterSets);

            return GenerateCandidateParameterSets(parameterSetCount, validParameterSets.ToList(), model);
        }

        RegressionForestModel FitModel(IEnumerable<OptimizerResult> validParameterSets)
        {
            var observations = validParameterSets
                .Select(v => v.ParameterSet).ToList()
                .ToF64Matrix();

            var targets = validParameterSets
                .Select(v => v.Error).ToArray();

            return m_learner.Learn(observations, targets);
        }

        double[][] GenerateCandidateParameterSets(int parameterSetCount,
            IReadOnlyList<OptimizerResult> previousResults, RegressionForestModel model)
        {
            // TODO: Handle maximization and minimization. Currently minimizes.
            var best = previousResults.Min(v => v.Error);

            // Sample new candidates.
            var results = FindNextCandidates(model, best);

            // Return the top candidate sets requested.
            // Error is used to store ExpectedImprovement, so we want the maximum value
            // not the minimum.
            var candidates = results
                .Where(v => !double.IsNaN(v.Error))
                .OrderByDescending(r => r.Error) 
                .Take(parameterSetCount)
                .Select(p => p.ParameterSet).ToArray();

            return candidates;
        }

        OptimizerResult[] FindNextCandidates(RegressionForestModel model, double bestScore)
        {
            // Additional set of random parameterSets to choose from during local search.
            var results = new List<OptimizerResult>();
            for (int i = 0; i < m_randomSearchPointCount; i++)
            {
                var parameterSet = RandomSearchOptimizer
                    .SampleParameterSet(m_parameters, m_sampler);

                var expectedImprovement = ComputeExpectedImprovement(bestScore, parameterSet, model);
                results.Add(new OptimizerResult(parameterSet, expectedImprovement));
            }

            return results.ToArray();
        }

        double ComputeExpectedImprovement(double best, double[] parameterSet, RegressionForestModel model)
        {
            var prediction = model.PredictCertainty(parameterSet);
            var mean = prediction.Prediction;
            var variance = prediction.Variance;
            return AcquisitionFunctions.ExpectedImprovement(best, mean, variance);
        }
    }
}