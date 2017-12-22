using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Containers.Matrices;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Models;
using SharpLearning.Containers.Extensions;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Sequential Model-based optimization (SMBO). SMBO learns a model based on the initial parameter sets and scores.
    /// This model is used to sample new promising parameter candiates which are evaluated and added to the existing paramter sets.
    /// This process iterates several times. The method is computational expensive so is most relevant for expensive problems, 
    /// where each evaluation of the function to minimize takes a long time, like hyper parameter tuning a machine learning method.
    /// But in that case it can usually reduce the number of iterations required to reach a good solution compared to less sophisticated methods.
    /// Implementation loosely based on:
    /// http://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf
    /// https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
    /// https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
    /// </summary>
    public sealed class SequentialModelBasedOptimizer : IOptimizer
    {
        readonly double[][] m_parameters;
        readonly int m_maxIterations;
        readonly int m_numberOfStartingPoints;
        readonly int m_numberOfCandidatesEvaluatedPrIteration;
        readonly Random m_random;

        readonly List<double[]> m_previousParameterSets;
        readonly List<double> m_previousParameterSetScores;

        // Important to use extra trees learner to have split bestween features calculated as: 
        // m_random.NextDouble() * (max - min) + min; 
        // instead of: (currentValue + prevValue) * 0.5; like in random forest.
        readonly RegressionExtremelyRandomizedTreesLearner m_learner;
        readonly RandomSearchOptimizer m_optimizer;

        /// <summary>
        /// Sequential Model-based optimization (SMBO). SMBO learns a model based on the initial parameter sets and scores.
        /// This model is used to sample new promising parameter candiates which are evaluated and added to the existing paramter sets.
        /// This process iterates several times. The method is computational expensive so is most relevant for expensive problems, 
        /// where each evaluation of the function to minimize takes a long time, like hyper parameter tuning a machine learning method.
        /// But in that case it can usually reduce the number of iterations required to reach a good solution compared to less sophisticated methods.
        /// Implementation loosely based on:
        /// http://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf
        /// https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
        /// https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
        /// </summary>
        /// <param name="parameters">Each row is a series of values for a specific parameter</param>
        /// <param name="maxIterations">Maximum number of iterations. MaxIteration * numberOfCandidatesEvaluatedPrIteration = totalFunctionEvaluations</param>
        /// <param name="numberOfStartingPoints">Number of randomly created starting points to use for the initial model in the first iteration (default is 10)</param>
        /// <param name="numberOfCandidatesEvaluatedPrIteration">How many candiate parameter set should by sampled from the model in each iteration. 
        /// The parameter sets are inlcuded in order of most promissing outcome (default is 3)</param>
        /// <param name="seed">Seed for the random initialization</param>
        public SequentialModelBasedOptimizer(double[][] parameters, int maxIterations, int numberOfStartingPoints = 10, int numberOfCandidatesEvaluatedPrIteration = 3, int seed = 42)
        {
            if (parameters == null) { throw new ArgumentNullException("parameters"); }
            if (maxIterations <= 0) { throw new ArgumentNullException("maxIterations must be at least 1"); }
            if (numberOfStartingPoints < 1) { throw new ArgumentNullException("numberOfParticles must be at least 1"); }

            m_parameters = parameters;
            m_maxIterations = maxIterations;
            m_numberOfStartingPoints = numberOfStartingPoints;
            m_numberOfCandidatesEvaluatedPrIteration = numberOfCandidatesEvaluatedPrIteration;

            m_random = new Random(seed);
            
            // Hyper parameters for regression extra trees learner. These are based on the values suggested in http://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf.
            // However, according to the author Frank Hutter, the hyper parameters for the forest model should not matter much.
            m_learner = new RegressionExtremelyRandomizedTreesLearner(10, 10, 2000, parameters.Length, 1e-6, 1.0, 42, false);

            // optimizer for finding maximum expectation (most promissing hyper parameters) from extra trees model.
            m_optimizer = new RandomSearchOptimizer(m_parameters, 1000, 42, false);
        }


        /// <summary>
        /// Sequential Model-based optimization (SMBO). SMBO learns a model based on the initial parameter sets and scores.
        /// This model is used to sample new promising parameter candiates which are evaluated and added to the existing paramter sets.
        /// This process iterates several times. The method is computational expensive so is most relevant for expensive problems, 
        /// where each evaluation of the function to minimize takes a long time, like hyper parameter tuning a machine learning method.
        /// But in that case it can usually reduce the number of iterations required to reach a good solution compared to less sophisticated methods.
        /// Implementation loosely based on:
        /// http://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf
        /// https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
        /// https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
        /// </summary>
        /// <param name="parameters">Each row is a series of values for a specific parameter</param>
        /// <param name="maxIterations">Maximum number of iterations. MaxIteration * numberOfCandidatesEvaluatedPrIteration = totalFunctionEvaluations</param>
        /// <param name="previousParameterSets">Parameter sets from previous run</param>
        /// <param name="previousParameterSetScores">Scores from from previous run corresponding to each parameter set</param>
        /// <param name="numberOfCandidatesEvaluatedPrIteration">How many candiate parameter set should by sampled from the model in each iteration. 
        /// The parameter sets are inlcuded in order of most promissing outcome (default is 3)</param>
        /// <param name="seed">Seed for the random initialization</param>
        public SequentialModelBasedOptimizer(double[][] parameters, int maxIterations, List<double[]> previousParameterSets, List<double> previousParameterSetScores,
            int numberOfCandidatesEvaluatedPrIteration = 3, int seed = 42)
        {
            if (parameters == null) { throw new ArgumentNullException("parameters"); }
            if (maxIterations <= 0) { throw new ArgumentNullException("maxIterations must be at least 1"); }
            if (previousParameterSets == null) { throw new ArgumentNullException("previousParameterSets"); }
            if (previousParameterSetScores == null) { throw new ArgumentNullException("previousResults"); }
            if (previousParameterSets.Count != previousParameterSetScores.Count) { throw new ArgumentException("previousParameterSets length: " 
                + previousParameterSets.Count + " does not correspond with previousResults length: " + previousParameterSetScores.Count); }
            if (previousParameterSetScores.Count < 2 || previousParameterSets.Count < 2)
            { throw new ArgumentException("previousParameterSets length and previousResults length must be at least 2 and was: " + previousParameterSetScores.Count); }


            m_parameters = parameters;
            m_maxIterations = maxIterations;
            m_numberOfCandidatesEvaluatedPrIteration = numberOfCandidatesEvaluatedPrIteration;

            m_random = new Random(seed);

            // Hyper parameters for regression extra trees learner. These are based on the values suggested in http://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf.
            // However, according to the author Frank Hutter, the hyper parameters for the forest model should not matter much.
            m_learner = new RegressionExtremelyRandomizedTreesLearner(10, 10, 2000, parameters.Length, 1e-6, 1.0, 42, false);

            // optimizer for finding maximum expectation (most promissing hyper parameters) from random forest model
            m_optimizer = new RandomSearchOptimizer(m_parameters, 1000, 42, false);

            m_previousParameterSets = previousParameterSets;
            m_previousParameterSetScores = previousParameterSetScores;
        }

        /// <summary>
        /// Optimization using Sequential Model-based optimization.
        /// Returns the result which best minimises the provided function.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult OptimizeBest(Func<double[], OptimizerResult> functionToMinimize)
        {
            return Optimize(functionToMinimize).First();
        }

        /// <summary>
        /// Optimization using Sequential Model-based optimization.
        /// Returns the final results ordered from best to worst (minimized).
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult[] Optimize(Func<double[], OptimizerResult> functionToMinimize)
        {
            var bestParameterSet = new double[m_parameters.Length];
            var bestParameterSetScore = double.MaxValue;

            // initialize max and min parameter bounds
            var maxParameters = new double[m_parameters.Length];
            var minParameters = new double[m_parameters.Length];
            for (int i = 0; i < m_parameters.Length; i++)
            {
                maxParameters[i] = m_parameters[i].Max();
                minParameters[i] = m_parameters[i].Min();
            }

            var parameterSets = new List<double[]>();
            var parameterSetScores = new List<double>();

            var usePreviousResults = m_previousParameterSetScores != null && m_previousParameterSets != null;

            int interations = 0;

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
                    var set = CreateParameterSet();
                    var score = functionToMinimize(set).Error;
                    interations++;

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
                        parameterSet = CreateParameterSet();
                    }

                    var result = functionToMinimize(parameterSet);
                    interations++;

                    if (!double.IsNaN(result.Error))
                    {
                        // update best
                        if (result.Error < bestParameterSetScore)
                        {
                            bestParameterSetScore = result.Error;
                            bestParameterSet = result.ParameterSet;
                            //Console.WriteLine("New Best: " + result.Error + " : " + string.Join(", ", result.ParameterSet));
                        }

                        // add point to parameter set list for next iterations model
                        parameterSets.Add(result.ParameterSet);
                        parameterSetScores.Add(result.Error);

                        System.Diagnostics.Trace.WriteLine(iteration + ";" + result.Error);
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

            return results.Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).ToArray();
        }

        OptimizerResult[] FindNextCandidates(RegressionForestModel model, double bestScore)
        {
            Func<double[], OptimizerResult> minimize = (param) =>
            {
                // use the model to predict the expected performance, mean and variance, of the parameter set.
                var p = model.PredictCertainty(param);

                return new OptimizerResult(param,
                    // negative, since we want to "maximize" the acquisition function.
                    -AcquisitionFunctions.ExpectedImprovement(bestScore, p.Prediction, p.Variance));
            };

            return m_optimizer.Optimize(minimize).Take(m_numberOfCandidatesEvaluatedPrIteration).ToArray();
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

        double[] CreateParameterSet()
        {
            var newPoint = new double[m_parameters.Length];

            for (int i = 0; i < m_parameters.Length; i++)
            {
                var range = m_parameters[i];
                newPoint[i] = NewParameter(range.Min(), range.Max());
            }

            return newPoint;
        }

        double NewParameter(double min, double max)
        {
            return m_random.NextDouble() * (max - min) + min;
        }
    }
}
