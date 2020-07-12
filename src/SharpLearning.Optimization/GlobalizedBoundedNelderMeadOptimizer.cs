using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using SharpLearning.Containers.Arithmetic;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Globalized bounded Nelder-Mead method. This version of Nelder-Mead optimization 
    /// avoids some of the shortcomings the standard implementation. 
    /// Specifically it is better suited for multi-modal optimization problems through its restart property.
    /// It also respect the bounds given by the provided parameter space.
    /// Roughly based on:
    /// http://home.ku.edu.tr/~daksen/2004-Nelder-Mead-Method-Wolff.pdf
    /// and
    /// http://www.emse.fr/~leriche/GBNM_SMO_1026_final.pdf
    /// </summary>
    public sealed class GlobalizedBoundedNelderMeadOptimizer : IOptimizer
    {
        readonly int m_maxIterationsPrRestart;
        readonly int m_maxIterationsWithoutImprovement;
        readonly int m_maxRestarts;
        readonly double m_alpha;
        readonly double m_gamma;
        readonly double m_rho;
        readonly double m_sigma;
        readonly double m_noImprovementThreshold;
        readonly IParameterSpec[] m_parameters;
        readonly Random m_random;
        readonly IParameterSampler m_sampler;
        readonly int m_maxFunctionEvaluations;
        int m_totalFunctionEvaluations;
        readonly int m_maxDegreeOfParallelism = -1;

        /// <summary>
        /// Globalized bounded Nelder-Mead method. This version of Nelder-Mead optimization 
        /// avoids some of the shortcomings the standard implementation. 
        /// Specifically it is better suited for multi-modal optimization problems through its restart property.
        /// It also respect the bounds given by the provided parameter space.
        /// Roughly based on:
        /// http://home.ku.edu.tr/~daksen/2004-Nelder-Mead-Method-Wolff.pdf
        /// and
        /// http://www.emse.fr/~leriche/GBNM_SMO_1026_final.pdf
        /// </summary>
        /// <param name="parameters">A list of parameter specs, one for each optimization parameter</param>
        /// <param name="maxRestarts">Maximum number of restart (default is 8</param>
        /// <param name="noImprovementThreshold">Minimum value of improvement before the improvement is accepted as an actual improvement (default is 0.001)</param>
        /// <param name="maxIterationsWithoutImprovement">Maximum number of iterations without an improvement (default is 5)</param>
        /// <param name="maxIterationsPrRestart">Maximum iterations pr. restart. 0 is no limit and will run to convergence (default is 0)</param>
        /// <param name="maxFunctionEvaluations">Maximum function evaluations. 0 is no limit and will run to convergence (default is 0)</param>
        /// <param name="alpha">Coefficient for reflection part of the algorithm (default is 1)</param>
        /// <param name="gamma">Coefficient for expansion part of the algorithm (default is 2)</param>
        /// <param name="rho">Coefficient for contraction part of the algorithm (default is -0.5)</param>
        /// <param name="sigma">Coefficient for shrink part of the algorithm (default is 0.5)</param>
        /// <param name="seed">Seed for random restarts</param>
        /// <param name="maxDegreeOfParallelism">Maximum number of concurrent operations (default is -1 (unlimited))</param>
        public GlobalizedBoundedNelderMeadOptimizer(IParameterSpec[] parameters, 
            int maxRestarts=8, 
            double noImprovementThreshold = 0.001, 
            int maxIterationsWithoutImprovement = 5, 
            int maxIterationsPrRestart = 0, 
            int maxFunctionEvaluations = 0,
            double alpha = 1, 
            double gamma = 2, 
            double rho = -0.5, 
            double sigma = 0.5, 
            int seed = 324, 
            int maxDegreeOfParallelism = -1)
        {
            if (maxIterationsWithoutImprovement <= 0) { throw new ArgumentException("maxIterationsWithoutImprovement must be at least 1"); }
            if (maxFunctionEvaluations < 0) { throw new ArgumentException("maxFunctionEvaluations must be at least 1"); }

            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            m_maxRestarts = maxRestarts;
            m_maxIterationsPrRestart = maxIterationsPrRestart;
            m_alpha = alpha;
            m_gamma = gamma;
            m_rho = rho;
            m_sigma = sigma;
            m_noImprovementThreshold = noImprovementThreshold;
            m_maxIterationsWithoutImprovement = maxIterationsWithoutImprovement;
            m_maxFunctionEvaluations = maxFunctionEvaluations;
            m_maxDegreeOfParallelism = maxDegreeOfParallelism;

            m_random = new Random(seed);

            // Use member to seed the random uniform sampler.
            m_sampler = new RandomUniform(m_random.Next());
        }
        /// <summary>
        /// Optimization using Globalized bounded Nelder-Mead method.
        /// Returns the result which best minimizes the provided function.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult OptimizeBest(Func<double[], OptimizerResult> functionToMinimize) =>
            // Return the best model found.
            Optimize(functionToMinimize).Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).First();

        /// <summary>
        /// Optimization using Globalized bounded Nelder-Mead method.
        /// Returns all results, chronologically ordered. 
        /// Note that the order of results might be affected if running parallel.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult[] Optimize(Func<double[], OptimizerResult> functionToMinimize)
        {
            var dim = m_parameters.Length;
            var initialPoint = new double[dim];

            var allResults = new List<OptimizerResult>();

            m_totalFunctionEvaluations = 0;

            for (int restarts = 0; restarts < m_maxRestarts; restarts++)
            {
                RandomRestartPoint(initialPoint);

                var previousBest = EvaluateFunction(functionToMinimize, initialPoint);
                var iterationsWithoutImprovement = 0;
                var concurrentResults = new ConcurrentDictionary<int, OptimizerResult>();

                var options = new ParallelOptions { MaxDegreeOfParallelism = m_maxDegreeOfParallelism };

                Parallel.For(0, dim, options, (i) =>
                {
                    var a = (0.02 + 0.08 * m_random.NextDouble()) * (m_parameters[i].Max - m_parameters[i].Min); // % simplex size between 2%-8% of min(xrange)

                    var p = a * (Math.Sqrt(dim + 1) + dim - 1) / (dim * Math.Sqrt(2));
                    var q = a * (Math.Sqrt(dim + 1) - 1) / (dim * Math.Sqrt(2));

                    var x = initialPoint.ToArray();
                    x[i] = x[i] + p;

                    for (int j = 0; j < dim; j++)
                    {
                        if (j != i)
                        {
                            x[j] = x[j] + q;
                        }
                    }

                    BoundCheck(x);
                    var result = EvaluateFunction(functionToMinimize, x);
                    concurrentResults.AddOrUpdate(i, result, (index, r) => r);
                });

                var results = concurrentResults.Values.ToList();
                // Add previous best result for simplex iteration.
                results.Add(previousBest);

                // simplex iter
                var iterations = 0;
                while (true)
                {
                    results = results.OrderBy(r => r.Error).ToList();
                    var best = results.First();

                    // break after m_maxIterationsPrRestart
                    if (iterations >= m_maxIterationsPrRestart && m_maxIterationsPrRestart != 0)
                    {
                        allResults.AddRange(results);
                        break;
                    }

                    iterations++;

                    if (best.Error < (previousBest.Error - m_noImprovementThreshold))
                    {
                        iterationsWithoutImprovement = 0;
                        previousBest = best;
                    }
                    else
                    {
                        iterationsWithoutImprovement++;
                    }

                    // break after no_improv_break iterations with no improvement
                    if (iterationsWithoutImprovement >= m_maxIterationsWithoutImprovement)
                    {
                        allResults.AddRange(results);
                        break;
                    }

                    // check if m_maxFunctionEvaluations is reached
                    if (m_totalFunctionEvaluations >= m_maxFunctionEvaluations && m_maxFunctionEvaluations != 0)
                    {
                        allResults.AddRange(results);
                        break;
                    }

                    // centroid
                    var x0 = new double[dim];

                    foreach (var tup in results.Take(results.Count - 1))
                    {
                        var parameters = tup.ParameterSet;
                        for (int i = 0; i < parameters.Length; i++)
                        {
                            x0[i] += parameters[i] / (results.Count - 1);
                        }
                    }

                    BoundCheck(x0);

                    // reflection
                    var last = results.Last();
                    var xr = x0.Add(x0.Subtract(last.ParameterSet).Multiply(m_alpha));
                    BoundCheck(xr);
                    var refelctionScore = EvaluateFunction(functionToMinimize, xr);

                    var first = results.First().Error;
                    if (first <= refelctionScore.Error && refelctionScore.Error < results[results.Count - 2].Error)
                    {
                        results.RemoveAt(results.Count - 1);
                        results.Add(refelctionScore);
                        continue;
                    }

                    // expansion
                    if (refelctionScore.Error < first)
                    {
                        var xe = x0.Add(x0.Subtract(last.ParameterSet).Multiply(m_gamma));
                        BoundCheck(xe);
                        var expansionScore = EvaluateFunction(functionToMinimize, xe);
                        if (expansionScore.Error < refelctionScore.Error)
                        {
                            results.RemoveAt(results.Count - 1);
                            results.Add(expansionScore);
                            continue;
                        }
                        else
                        {
                            results.RemoveAt(results.Count - 1);
                            results.Add(refelctionScore);
                            continue;
                        }
                    }

                    // contraction
                    var xc = x0.Add(x0.Subtract(last.ParameterSet).Multiply(m_rho));
                    BoundCheck(xc);
                    var contractionScore = EvaluateFunction(functionToMinimize, xc);
                    if (contractionScore.Error < last.Error)
                    {
                        results.RemoveAt(results.Count - 1);
                        results.Add(contractionScore);
                        continue;
                    }

                    // shrink
                    var x1 = results[0].ParameterSet;
                    var nres = new List<OptimizerResult>();
                    foreach (var tup in results)
                    {
                        var xs = x1.Add(x1.Subtract(tup.ParameterSet).Multiply(m_sigma));
                        BoundCheck(xs);
                        var score = EvaluateFunction(functionToMinimize, xs);
                        nres.Add(score);
                    }

                    results = nres.ToList();
                }

                // check if m_maxFunctionEvaluations is reached
                if (m_totalFunctionEvaluations >= m_maxFunctionEvaluations && m_maxFunctionEvaluations != 0)
                {
                    allResults.AddRange(results);
                    break;
                }
            }

            return allResults.ToArray();
        }

        OptimizerResult EvaluateFunction(Func<double[], OptimizerResult> functionToMinimize, 
            double[] parameters)
        {
            m_totalFunctionEvaluations++;
            return functionToMinimize(parameters);
        }

        /// <summary>
        /// Make sure the parameter set is within the specified bounds
        /// </summary>
        /// <param name="parameters"></param>
        void BoundCheck(double[] parameters)
        {
            for (int i = 0; i < parameters.Length; i++)
            {
                var parameter = m_parameters[i];
                parameters[i] = Math.Max(parameter.Min, Math.Min(parameters[i], parameter.Max));
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="newPoint"></param>
        void RandomRestartPoint(double[] newPoint)
        {
            // consider to implement Gaussian selection of next point to avoid very similar point being selected
            // look at: https://github.com/ojdo/gbnm/blob/master/gbnm.m
            for (int i = 0; i < m_parameters.Length; i++)
            {
                var parameter = m_parameters[i];
                newPoint[i] = parameter.SampleValue(m_sampler);
            }
        }
    }
}
