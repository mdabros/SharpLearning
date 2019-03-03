using System;
using System.Collections.Generic;
using System.Linq;
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
        readonly int m_maxIterations;
        readonly int m_numberOfStartingPoints;
        readonly int m_numberOfCandidatesEvaluatedPrIteration;
        readonly IParameterSampler m_sampler;
        readonly Random m_random;

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
        /// <param name="maxIterations">Maximum number of iterations. MaxIteration * numberOfCandidatesEvaluatedPrIteration = totalFunctionEvaluations</param>
        /// <param name="numberOfStartingPoints">Number of randomly created starting points to use for the initial model in the first iteration (default is 5)</param>
        /// <param name="numberOfCandidatesEvaluatedPrIteration">How many candidate parameter set should by sampled from the model in each iteration. 
        /// The parameter sets are included in order of most promising outcome (default is 1)</param>
        /// <param name="seed">Seed for the random initialization</param>
        public BayesianOptimizer(IParameterSpec[] parameters, int maxIterations, 
            int numberOfStartingPoints = 5, int numberOfCandidatesEvaluatedPrIteration = 1, int seed = 42)
        {
            if (maxIterations <= 0) { throw new ArgumentException("maxIterations must be at least 1"); }
            if (numberOfStartingPoints < 1) { throw new ArgumentException("numberOfParticles must be at least 1"); }

            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            m_maxIterations = maxIterations;
            m_numberOfStartingPoints = numberOfStartingPoints;
            m_numberOfCandidatesEvaluatedPrIteration = numberOfCandidatesEvaluatedPrIteration;

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
        /// <param name="maxIterations">Maximum number of iterations. MaxIteration * numberOfCandidatesEvaluatedPrIteration = totalFunctionEvaluations</param>
        /// <param name="previousParameterSets">Parameter sets from previous run</param>
        /// <param name="previousParameterSetScores">Scores from previous run corresponding to each parameter set</param>
        /// <param name="numberOfCandidatesEvaluatedPrIteration">How many candidate parameter set should by sampled from the model in each iteration. 
        /// The parameter sets are included in order of most promising outcome (default is 1)</param>
        /// <param name="seed">Seed for the random initialization</param>
        public BayesianOptimizer(IParameterSpec[] parameters, int maxIterations, 
            List<double[]> previousParameterSets, List<double> previousParameterSetScores,
            int numberOfCandidatesEvaluatedPrIteration = 1, int seed = 42)
        {
            if (maxIterations <= 0) { throw new ArgumentNullException("maxIterations must be at least 1"); }
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

            m_maxIterations = maxIterations;
            m_numberOfCandidatesEvaluatedPrIteration = numberOfCandidatesEvaluatedPrIteration;

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
        /// Optimization using Sequential Model-based optimization.
        /// Returns all results, chronologically ordered. 
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult[] Optimize(Func<double[], OptimizerResult> functionToMinimize)
        {
            var bestParameterSet = new double[m_parameters.Length];
            var bestParameterSetScore = double.MaxValue;

            var parameterSets = new List<double[]>();
            var parameterSetScores = new List<double>();

            var usePreviousResults = m_previousParameterSetScores != null && m_previousParameterSets != null;

            int iterations = 0;

            if (usePreviousResults)
            {
                parameterSets.AddRange(m_previousParameterSets);
                parameterSetScores.AddRange(m_previousParameterSetScores);

                for (int i = 0; i < parameterSets.Count; i++)
                {
                    var score = parameterSetScores[i];
                    if (!double.IsNaN(score))
                    {
                        if (score < bestParameterSetScore)
                        {
                            bestParameterSetScore = score;
                            bestParameterSet = parameterSets[i];
                        }
                    }
                }
            }
            else
            {
                // initialize random starting points for the first iteration
                for (int i = 0; i < m_numberOfStartingPoints; i++)
                {
                    var set = RandomSearchOptimizer.SampleParameterSet(m_parameters, m_sampler);
                    var score = functionToMinimize(set).Error;
                    iterations++;

                    if (!double.IsNaN(score))
                    {
                        parameterSets.Add(set);
                        parameterSetScores.Add(score);

                        if (score < bestParameterSetScore)
                        {
                            bestParameterSetScore = score;
                            bestParameterSet = set;
                        }
                    }
                }
            }

            var lastSet = new double[m_parameters.Length];
            for (int iteration = 0; iteration < m_maxIterations; iteration++)
            {
                // fit model
                var observations = parameterSets.ToF64Matrix();
                var targets = parameterSetScores.ToArray();
                var model = m_learner.Learn(observations, targets);

                var bestScore = parameterSetScores.Min();
                var candidates = FindNextCandidates(model, bestScore);
                
                var first = true;

                foreach (var candidate in candidates)
                {
                    var parameterSet = candidate.ParameterSet;

                    if (Equals(lastSet, parameterSet) && !first)
                    {
                        // skip evaluation if parameters have not changed.
                        continue;
                    }

                    if (Equals(bestParameterSet, parameterSet))
                    {
                        // if the beset parameter set is sampled again.
                        // Add a new random parameter set.
                        parameterSet = RandomSearchOptimizer
                            .SampleParameterSet(m_parameters, m_sampler);
                    }

                    var result = functionToMinimize(parameterSet);
                    iterations++;

                    if (!double.IsNaN(result.Error))
                    {
                        // update best
                        if (result.Error < bestParameterSetScore)
                        {
                            bestParameterSetScore = result.Error;
                            bestParameterSet = result.ParameterSet;
                            //System.Diagnostics.Trace.WriteLine(iterations + ";" + result.Error);
                        }

                        // add point to parameter set list for next iterations model
                        parameterSets.Add(result.ParameterSet);
                        parameterSetScores.Add(result.Error);                       
                    }

                    lastSet = parameterSet;
                    first = false;
                }
            }

            var results = new List<OptimizerResult>();

            for (int i = 0; i < parameterSets.Count; i++)
            {
                results.Add(new OptimizerResult(parameterSets[i], parameterSetScores[i]));
            }

            return results.ToArray();
        }

        OptimizerResult[] FindNextCandidates(RegressionForestModel model, double bestScore)
        {
            Func<double[], OptimizerResult> minimize = (param) =>
            {
                // use the model to predict the expected performance, mean and variance, of the parameter set.
                var p = model.PredictCertainty(param);

                return new OptimizerResult(param,
                    // negative, since we want to "maximize" the acquisition function.
                    -m_acquisitionFunc(bestScore, p.Prediction, p.Variance));
            };

            return m_maximizer.Optimize(minimize)
                .Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error)
                .Take(m_numberOfCandidatesEvaluatedPrIteration).ToArray();
        }

        bool Equals(double[] p1, double[] p2)
        {
            for (int i = 0; i < p1.Length; i++)
            {
                if (!Equal(p1[i], p2[i]))
                {
                    return false;
                }
            }

            return true;
        }

        const double m_tolerence = 0.00001;

        bool Equal(double a, double b)
        {
            var diff = Math.Abs(a * m_tolerence);
            if (Math.Abs(a - b) <= diff)
            {
                return true;
            }

            return false;
        }
    }
}
