using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Containers.Extensions;
using SharpLearning.Optimization.ParameterSamplers;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Models;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Implementation of the SMAC algorithm for hyperparameter optimization.
    /// Based on: Sequential Model-Based Optimization for General Algorithm Configuration:
    /// https://ml.informatik.uni-freiburg.de/papers/11-LION5-SMAC.pdf
    /// Uses Bayesian optimization in tandem with a greedy local search on the top performing solutions.
    /// </summary>
    public class SmacOptimizer : IOptimizer
    {
        readonly Random m_random;
        readonly IParameterSampler m_sampler;
        readonly IParameterSpec[] m_parameters;
        readonly int m_iterations;
        readonly int m_randomStartingPointsCount;
        readonly int m_functionEvaluationsPerIterationCount;
        readonly int m_localSearchPointCount;
        readonly int m_randomSearchPointCount;

        // Important to use extra trees learner to have split between features calculated as: 
        // m_random.NextDouble() * (max - min) + min; 
        // instead of: (currentValue + prevValue) * 0.5; like in random forest.
        readonly RegressionExtremelyRandomizedTreesLearner m_learner;

        /// <summary>
        /// Implementation of the SMAC algorithm for hyperparameter optimization.
        /// Based on: Sequential Model-Based Optimization for General Algorithm Configuration:
        /// https://ml.informatik.uni-freiburg.de/papers/11-LION5-SMAC.pdf
        /// Uses Bayesian optimization in tandem with a greedy local search on the top performing solutions.
        /// </summary>
        /// <param name="parameters">A list of parameter specs, one for each optimization parameter</param>
        /// <param name="iterations">The number of iterations to perform.
        /// Iteration * functionEvaluationsPerIteration = totalFunctionEvaluations</param>
        /// <param name="randomStartingPointCount">Number of randomly parameter sets used 
        /// for initialization (default is 20)</param>
        /// <param name="functionEvaluationsPerIterationCount">The number of function evaluations per iteration. 
        /// The parameter sets are included in order of most promising outcome (default is 1)</param>
        /// <param name="localSearchPointCount">The number of top contenders 
        /// to use in the greedy local search (default is (10)</param>
        /// <param name="randomSearchPointCount">The number of random parameter sets
        /// used when maximizing the expected improvement acquisition function (default is 1000)</param>
        /// <param name="seed"></param>
        public SmacOptimizer(IParameterSpec[] parameters,
            int iterations,
            int randomStartingPointCount = 20,
            int functionEvaluationsPerIterationCount = 1,
            int localSearchPointCount = 10,
            int randomSearchPointCount = 1000,
            int seed = 42)
        {
            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));

            m_random = new Random(seed);
            // Use member to seed the random uniform sampler.
            m_sampler = new RandomUniform(m_random.Next());
            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            m_iterations = iterations;
            m_randomStartingPointsCount = randomStartingPointCount;
            m_functionEvaluationsPerIterationCount = functionEvaluationsPerIterationCount;
            m_localSearchPointCount = localSearchPointCount;
            m_randomSearchPointCount = randomSearchPointCount;

            // Hyper parameters for regression extra trees learner. 
            // These are based on the values suggested in http://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf.
            // However, according to the author Frank Hutter, 
            // the hyper parameters for the forest model should not matter that much.
            m_learner = new RegressionExtremelyRandomizedTreesLearner(trees: 10,
                minimumSplitSize: 2,
                maximumTreeDepth: 2000,
                featuresPrSplit: parameters.Length,
                minimumInformationGain: 1e-6,
                subSampleRatio: 1.0,
                seed: m_random.Next(), // Use member to seed the random uniform sampler.
                runParallel: false);
        }

        public OptimizerResult OptimizeBest(Func<double[], OptimizerResult> functionToMinimize) =>
            // Return the best model found.
            Optimize(functionToMinimize).Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).First();

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

            // return all results ordered
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
        /// <param name="previousRuns">Results from previous runs.  
        /// These are used in the model for proposing new parameter sets.
        /// If no results are provided, random parameter sets will be returned.</param>
        /// <returns></returns>
        public double[][] ProposeParameterSets(int parameterSetCount, 
            IReadOnlyList<OptimizerResult> previousRuns = null)
        {
            var previousParameterSetCount = previousRuns == null ? 0 : previousRuns.Count;
            if (previousParameterSetCount < m_randomStartingPointsCount)
            {
                var randomParameterSetCount = Math.Min(parameterSetCount,
                    m_randomStartingPointsCount - previousParameterSetCount);

                var randomParameterSets = RandomSearchOptimizer.SampleRandomParameterSets(
                    randomParameterSetCount, m_parameters, m_sampler);

                return randomParameterSets;
            }

            var validParameterSets = previousRuns.Where(v => !double.IsNaN(v.Error));            
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
            IReadOnlyList<OptimizerResult> previousRuns, RegressionForestModel model)
        {
            // Get top parameter sets from previous runs.
            var topParameterSets = previousRuns.OrderBy(v => v.Error)
                .Take(m_localSearchPointCount).Select(v => v.ParameterSet).ToArray();

            // Perform local search using the top parameter sets from previous run.
            var challengerCount = (int)Math.Ceiling(parameterSetCount / 2.0F);
            var challengers = GreedyPlusRandomSearch(topParameterSets, model,
                challengerCount, previousRuns);

            // Create random parameter sets.
            var randomParameterSetCount = parameterSetCount - challengers.Length;
            var randomChallengers = RandomSearchOptimizer.SampleRandomParameterSets(
                randomParameterSetCount, m_parameters, m_sampler);

            // Interleave challengers and random parameter sets.
            return InterLeaveModelBasedAndRandomParameterSets(challengers, randomChallengers);
        }

        double[][] InterLeaveModelBasedAndRandomParameterSets(double[][] challengers, 
            double[][] randomChallengers)
        {
            var finalParameterSets = new double[challengers.Length + randomChallengers.Length][];
            Array.Copy(challengers, 0, finalParameterSets, 0, challengers.Length);
            Array.Copy(randomChallengers, 0, finalParameterSets, challengers.Length, randomChallengers.Length);
            return finalParameterSets;
        }

        double[][] GreedyPlusRandomSearch(double[][] parentParameterSets, RegressionForestModel model, 
            int parameterSetCount, IReadOnlyList<OptimizerResult> previousRuns)
        {
            // TODO: Handle maximization and minimization. Currently minimizes.
            var best = previousRuns.Min(v => v.Error);

            var parameterSets = new List<(double[] parameterSet, double EI)>();
           
            // Perform local search.
            foreach (var parameterSet in parentParameterSets)
            {
                var bestParameterSet = LocalSearch(parentParameterSets, model, best, epsilon: 0.00001);
                parameterSets.Add(bestParameterSet);
            }

            // Additional set of random parameterSets to choose from during local search.
            for (int i = 0; i < m_randomSearchPointCount; i++)
            {
                var parameterSet = RandomSearchOptimizer
                    .SampleParameterSet(m_parameters, m_sampler);

                var ei = ComputeExpectedImprovement(best, parameterSet, model);
                parameterSets.Add((parameterSet, ei));
            }

            // Take the best parameterSets. Here we want the max expected improvement.
            return parameterSets.OrderByDescending(v => v.EI)
                .Take(parameterSetCount).Select(v => v.parameterSet)
                .ToArray();
        }

        /// <summary>
        /// Performs a local one-mutation neighborhood greedy search.
        /// Stop search when no neighbors increase expected improvement.
        /// </summary>
        (double[] parameterSet, double expectedImprovement) LocalSearch(double[][] parentParameterSets, 
            RegressionForestModel model, double bestScore, double epsilon)
        {
            var bestParameterSet = parentParameterSets.First();
            var bestExpectedImprovement = ComputeExpectedImprovement(bestScore, bestParameterSet, model);

            // Continue search until no improvement is found.
            var continueSearch = true;
            while (continueSearch)
            {
                continueSearch = false;
                var neighborhood = GetOneMutationNeighborhood(bestParameterSet);
                for (int i = 0; i < neighborhood.Count; i++)
                {
                    var neighbor = neighborhood[i];
                    var ei = ComputeExpectedImprovement(bestScore, neighbor, model);
                    if (ei - bestExpectedImprovement > epsilon)
                    {
                        bestParameterSet = neighbor;
                        bestExpectedImprovement = ei;
                        continueSearch = true;
                    }
                }
            }

            return (bestParameterSet, bestExpectedImprovement);
        }

        List<double[]> GetOneMutationNeighborhood(double[] parentParameterSet)
        {
            var neighbors = new List<double[]>();
            for (int i = 0; i < m_parameters.Length; i++)
            {
                // Add a new parameter set that differs only by one parameter from the parent.
                var parameterSpec = m_parameters[i];

                // Add 4 parameterSets pr. parameter. 
                // This case if for continuous variables.
                // Original paper also has a case for categorical parameters.
                // However, this is currently not supported.
                const int parameterSetCount = 4;
                for (int j = 0; j < parameterSetCount; j++)
                {
                    // Copy parent and mutate one parameter.
                    var newParameterSet = parentParameterSet.ToArray();
                    newParameterSet[i] = parameterSpec.SampleValue(m_sampler);
                    neighbors.Add(newParameterSet);
                }
            }

            return neighbors;
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
