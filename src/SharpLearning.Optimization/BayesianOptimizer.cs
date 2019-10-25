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
        readonly IParameterSpec[] m_parameters;
        readonly int m_iterations;
        readonly int m_randomStartingPointsCount;
        readonly int m_functionEvaluationsPerIterationCount;
        private readonly bool m_runParallel;
        private readonly ParallelOptions m_parallelOptions;
        private readonly bool m_allowMultipleEvaluations;
        readonly IParameterSampler m_sampler;
        readonly Random m_random;
        readonly object m_locker;
        const double m_tolerence = 0.00001;

        readonly List<double[]> m_previousParameterSets;
        readonly List<double> m_previousParameterSetScores;

        // Important to use extra trees learner to have split between features calculated as: 
        // m_random.NextDouble() * (max - min) + min; 
        // instead of: (currentValue + prevValue) * 0.5; like in random forest.
        readonly RegressionExtremelyRandomizedTreesLearner m_learner;

        // Optimizer for finding maximum expectation (most promising hyper parameters) from extra trees model.
        readonly IOptimizer m_maximizer;

        // Acquisition function to maximize
        readonly AcquisitionFunction m_acquisitionFunc;
        private bool m_isFirst;

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
        /// <param name="iterations">Number of iterations. Iteration * functionEvaluationsPerIteration = totalFunctionEvaluations</param>
        /// <param name="randomStartingPointCount">Number of randomly created starting points to use for the initial model in the first iteration (default is 5)</param>
        /// <param name="functionEvaluationsPerIteration">The number of function evaluations per iteration. 
        /// The parameter sets are included in order of most promising outcome (default is 1)</param>
        /// <param name="seed">Seed for the random initialization</param>
        /// <param name="maxDegreeOfParallelism">Maximum number of concurrent operations. Default is -1 (unlimited)</param>
        /// <param name="allowMultipleEvaluations">Enables re-evaluation of duplicate parameter sets for non-deterministic functions</param>
        public BayesianOptimizer(IParameterSpec[] parameters,
            int iterations,
            int randomStartingPointCount = 5,
            int functionEvaluationsPerIteration = 1,
            int seed = 42,
            int maxDegreeOfParallelism = -1,
            bool allowMultipleEvaluations = false)
        {
            if (iterations <= 0) { throw new ArgumentException("maxIterations must be at least 1"); }
            if (randomStartingPointCount < 1) { throw new ArgumentException("numberOfParticles must be at least 1"); }

            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            m_iterations = iterations;
            m_randomStartingPointsCount = randomStartingPointCount;
            m_functionEvaluationsPerIterationCount = functionEvaluationsPerIteration;
            m_runParallel = maxDegreeOfParallelism != 1;
            m_parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };
            m_allowMultipleEvaluations = allowMultipleEvaluations;
            m_locker = new object();

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
                runParallel: m_runParallel);

            // Optimizer for finding maximum expectation (most promising hyper parameters) from extra trees model.
            m_maximizer = new RandomSearchOptimizer(m_parameters, iterations: 1000,
                seed: m_random.Next(), // Use member to seed the random uniform sampler.
                runParallel: maxDegreeOfParallelism > 1);

            // Acquisition function to maximize.
            m_acquisitionFunc = AcquisitionFunctions.ExpectedImprovement;
        }


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
        /// <param name="iterations">Maximum number of iterations. MaxIteration * numberOfCandidatesEvaluatedPrIteration = totalFunctionEvaluations</param>
        /// <param name="previousParameterSets">Parameter sets from previous run</param>
        /// <param name="previousParameterSetScores">Scores from previous run corresponding to each parameter set</param>
        /// <param name="functionEvaluationsPerIteration">How many candidate parameter set should by sampled from the model in each iteration. 
        /// The parameter sets are included in order of most promising outcome (default is 1)</param>
        /// <param name="seed">Seed for the random initialization</param>
        /// <param name="maxDegreeOfParallelism">Maximum number of concurrent operations. Default is -1 (unlimited)</param>
        /// <param name="allowMultipleEvaluations">Enables re-evaluation of duplicate parameter sets for non-deterministic functions</param>
        public BayesianOptimizer(IParameterSpec[] parameters,
            int iterations,
            List<double[]> previousParameterSets,
            List<double> previousParameterSetScores,
            int functionEvaluationsPerIteration = 1,
            int seed = 42,
            int maxDegreeOfParallelism = -1,
            bool allowMultipleEvaluations = false)
        {
            if (iterations <= 0) { throw new ArgumentNullException("iterations must be at least 1"); }
            if (previousParameterSets.Count != previousParameterSetScores.Count)
            {
                throw new ArgumentException("previousParameterSets length: "
                    + previousParameterSets.Count + " does not correspond with previousResults length: "
                    + previousParameterSetScores.Count);
            }

            if (previousParameterSetScores.Count < 2 || previousParameterSets.Count < 2)
            {
                throw new ArgumentException("previousParameterSets length and previousResults length must be at least 2 and was: "
                    + previousParameterSetScores.Count);
            }

            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            m_previousParameterSets = previousParameterSets ?? throw new ArgumentNullException(nameof(previousParameterSets));
            m_previousParameterSetScores = previousParameterSetScores ?? throw new ArgumentNullException(nameof(previousParameterSetScores));

            m_iterations = iterations;
            m_functionEvaluationsPerIterationCount = functionEvaluationsPerIteration;
            m_parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };
            m_runParallel = maxDegreeOfParallelism != 1;
            m_allowMultipleEvaluations = allowMultipleEvaluations;
            m_locker = new object();

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
                runParallel: m_runParallel);

            // Optimizer for finding maximum expectation (most promising hyper parameters) from extra trees model.
            m_maximizer = new RandomSearchOptimizer(m_parameters, iterations: 1000,
                seed: m_random.Next(), // Use member to seed the random uniform sampler.
                runParallel: m_runParallel);

            // Acquisition function to maximize.
            m_acquisitionFunc = AcquisitionFunctions.ExpectedImprovement;
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


        ///// <summary>
        ///// Optimization using Sequential Model-based optimization.
        ///// Returns all results, chronologically ordered. 
        ///// </summary>
        ///// <param name="functionToMinimize"></param>
        ///// <returns></returns>
        //public OptimizerResult[] Optimize(Func<double[], OptimizerResult> functionToMinimize)
        //{
        //    var parameterSets = new BlockingCollection<(double[] Parameters, double Error)>();
        //    var usePreviousResults = m_previousParameterSetScores != null && m_previousParameterSets != null;

        //    int iterations = 0;

        //    if (usePreviousResults)
        //    {
        //        for (int i = 0; i < m_previousParameterSets.Count; i++)
        //        {
        //            var score = m_previousParameterSetScores[i];
        //            if (!double.IsNaN(score))
        //            {
        //                parameterSets.Add((m_previousParameterSets[i], score));
        //            }
        //        }
        //    }
        //    else
        //    {
        //        // initialize random starting points for the first iteration
        //        Parallel.For(0, m_randomStartingPointsCount, m_parallelOptions, i =>
        //        {
        //            var set = RandomSearchOptimizer.SampleParameterSet(m_parameters, m_sampler);
        //            var score = functionToMinimize(set).Error;
        //            iterations++;

        //            if (!double.IsNaN(score))
        //            {
        //                parameterSets.Add((set, score));
        //            }
        //        });
        //    }
        //    for (int iteration = 0; iteration < m_iterations; iteration++)
        //    {
        //        // fit model			
        //        var observations = parameterSets.Select(s => s.Parameters).ToList().ToF64Matrix();
        //        var targets = parameterSets.Select(s => s.Error).ToArray();
        //        var model = m_learner.Learn(observations, targets);

        //        var bestScore = parameterSets.Min(m => m.Error);
        //        var candidates = FindNextCandidates(model, bestScore);

        //        m_isFirst = true;

        //        Parallel.ForEach(candidates, m_parallelOptions, candidate =>
        //        {
        //            var parameterSet = candidate.ParameterSet;

        //            // skip evaluation if parameters have not changed unless explicitly allowed
        //            if (m_allowMultipleEvaluations || IsFirstEvaluation() || !Contains(parameterSets, parameterSet))
        //            {

        //                if (!m_allowMultipleEvaluations && Equals(GetBestParameterSet(parameterSets), parameterSet))
        //                {
        //                    // if the best parameter set is sampled again.
        //                    // Add a new random parameter set.
        //                    parameterSet = RandomSearchOptimizer
        //                        .SampleParameterSet(m_parameters, m_sampler);
        //                }

        //                var result = functionToMinimize(parameterSet);
        //                iterations++;

        //                if (!double.IsNaN(result.Error))
        //                {
        //                    // add point to parameter set list for next iterations model
        //                    parameterSets.Add((parameterSet, result.Error));
        //                }

        //            }
        //        });

        //    }

        //    return parameterSets.Select(p => new OptimizerResult(p.Parameters, p.Error)).ToArray();
        //}

        bool IsFirstEvaluation()
        {
            lock (m_locker)
            {
                if (m_isFirst)
                {
                    m_isFirst = false;
                    return true;
                }

            }

            return m_isFirst;
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
            var results = new List<OptimizerResult>();
            foreach (var parameterSet in parameterSets)
            {
                // Get the current parameters for the current point
                var result = functionToMinimize(parameterSet);
                results.Add(result);
            }

            return results;
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

            var validParameterSets = previousResults.Where(v => !double.IsNaN(v.Error));
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

            // Use maximizer for sampling potential new candidates.
            var results = FindNextCandidates(model, best);

            // Return the top candidate sets requested.
            var candidates = results
                .Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error)
                .Take(parameterSetCount)
                .Select(p => p.ParameterSet).ToArray();

            return candidates;
        }

        OptimizerResult[] FindNextCandidates(RegressionForestModel model, double bestScore)
        {
            OptimizerResult minimize(double[] param)
            {
                // use the model to predict the expected performance, mean and variance, of the parameter set.
                var p = model.PredictCertainty(param);

                return new OptimizerResult(param,
                    // negative, since we want to "maximize" the acquisition function.
                    -m_acquisitionFunc(bestScore, p.Prediction, p.Variance));
            }

            return m_maximizer.Optimize(minimize);
        }

        bool Equals(double[] p1, double[] p2)
        {
            if (p1 == null)
            {
                return false;
            }

            for (int i = 0; i < p1.Length; i++)
            {
                if (!Equal(p1[i], p2[i]))
                {
                    return false;
                }
            }

            return true;
        }

        bool Contains(BlockingCollection<(double[] Parameters, double Error)> many, double[] single)
        {
            lock (m_locker)
            {
                return many.Any(m => Equals(m.Parameters, single));
            }
        }

        bool Equal(double a, double b)
        {
            var diff = Math.Abs(a * m_tolerence);
            if (Math.Abs(a - b) <= diff)
            {
                return true;
            }

            return false;
        }

        double[] GetBestParameterSet(BlockingCollection<(double[] Parameters, double Error)> parameterSets)
        {
            lock (m_locker)
            {
                return parameterSets.FirstOrDefault(f => f.Error == parameterSets.Min(p => p.Error)).Parameters;
            }
        }
    }
}