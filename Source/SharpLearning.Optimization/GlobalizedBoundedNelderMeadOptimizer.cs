using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Containers.Arithmetic;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Globalized bounded Nelder-Mead method. This version of Nelder-Mead optimization 
    /// avoids some of the shortcommings the standard implementation. 
    /// Specifically it is better suited for multimodal optimization problems through its restart property.
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
        readonly double[][] m_parameters;
        readonly Random m_random;

        /// <summary>
        /// Globalized bounded Nelder-Mead method. This version of Nelder-Mead optimization 
        /// avoids some of the shortcommings the standard implementation. 
        /// Specifically it is better suited for multimodal optimization problems through its restart property.
        /// It also respect the bounds given by the provided parameter space.
        /// Roughly based on:
        /// http://home.ku.edu.tr/~daksen/2004-Nelder-Mead-Method-Wolff.pdf
        /// and
        /// http://www.emse.fr/~leriche/GBNM_SMO_1026_final.pdf
        /// </summary>
        /// <param name="parameters">Each row is a series of values for a specific parameter</param>
        /// <param name="maxRestarts">Maximun number of restart (default is 5)</param>
        /// <param name="noImprovementThreshold">Minimum value of improvement before the improvement is accepted as an actual improvement (default is 10e-6)</param>
        /// <param name="maxIterationsWithoutImprovement">Maximum number of iterations without an improvement (default is 5)</param>
        /// <param name="maxIterationsPrRestart">Maximum iterations pr. restart. 0 is no limit and will run to convergens (default is 0)</param>
        /// <param name="alpha">Coefficient for reflection part of the algorithm (default is 1)</param>
        /// <param name="gamma">Coefficient for expansion part of the algorithm (default is 2)</param>
        /// <param name="rho">Coefficient for contraction part of the algorithm (default is -0.5)</param>
        /// <param name="sigma">Coefficient for shrink part of the algorithm (default is 0.5)</param>
        public GlobalizedBoundedNelderMeadOptimizer(double[][] parameters, int maxRestarts=5, double noImprovementThreshold = 10e-6, 
            int maxIterationsWithoutImprovement = 5, int maxIterationsPrRestart = 0, 
            double alpha = 1, double gamma = 2, double rho = -0.5, double sigma = 0.5)
        {
            if (parameters == null) { throw new ArgumentNullException("x_start"); }
            if (maxIterationsWithoutImprovement <= 0) { throw new ArgumentNullException("maxIterationsWithoutImprovement must be at least 1"); }

            m_maxRestarts = maxRestarts;
            m_maxIterationsPrRestart = maxIterationsPrRestart;
            m_alpha = alpha;
            m_gamma = gamma;
            m_rho = rho;
            m_sigma = sigma;
            m_parameters = parameters;
            m_noImprovementThreshold = noImprovementThreshold;
            m_maxIterationsWithoutImprovement = maxIterationsWithoutImprovement;

            m_random = new Random(324);
        }
        /// <summary>
        /// Optimization using Globalized bounded Nelder-Mead method.
        /// Returns the result which best minimises the provided function.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult OptimizeBest(Func<double[], OptimizerResult> functionToMinimize)
        {
            return Optimize(functionToMinimize).First();
        }

        /// <summary>
        /// Optimization using Globalized bounded Nelder-Mead method.
        /// Returns the final results ordered from best to worst (minimized).
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult[] Optimize(Func<double[], OptimizerResult> functionToMinimize)
        {
            var dim = m_parameters.Length;
            var initialPoint = new double[dim];

            var allResults = new List<OptimizerResult>();

            for (int restarts = 0; restarts < m_maxRestarts; restarts++)
            {
                RandomRestartPoint(initialPoint);

                var prevBest = functionToMinimize(initialPoint);
                var iterationsWithoutImprovement = 0;
                var results = new List<OptimizerResult> { new OptimizerResult(prevBest.ParameterSet, prevBest.Error) };

                for (int i = 0; i < dim; i++)
                {
                    var a = (0.02 + 0.08 * m_random.NextDouble()) * (m_parameters[i].Max() - m_parameters[i].Min()); // % simplex size between 2%-8% of min(xrange)

                    var p = a * (Math.Sqrt(dim + 1) + dim - 1) / (dim * Math.Sqrt(2));
                    var x = initialPoint.ToArray();
                    x[i] = x[i] + p;

                    // adding q seems to reduce the exploration of the algorithm
                    //var q = a * (Math.Sqrt(dim + 1) - 1) / (dim * Math.Sqrt(2));
                    //for (int j = 0; j < dim; j++)
                    //{
                    //    if(j != i)
                    //    {
                    //        x[j] = x[j] + q;
                    //    }
                    //}

                    BoundCheck(x);
                    var score = functionToMinimize(x);
                    results.Add(score);

                    //Console.WriteLine("Intials: " + score.Error + " " + string.Join(", ", score.ParameterSet));
                }

               // Console.WriteLine(restarts);

                // simplex iter
                var iterations = 0;
                while (true)
                {
                    results = results.OrderBy(r => r.Error).ToList();
                    var best = results.First();

                    // break after max_iter
                    if (iterations >= m_maxIterationsPrRestart && m_maxIterationsPrRestart != 0)
                    {
                        allResults.AddRange(results);
                        break;
                        //return results.ToArray();
                    }

                    iterations++;

                    // break after no_improv_break iterations with no improvement
                    //Console.WriteLine("Current best:" + best.Error + " " + string.Join(", ", best.ParameterSet));

                    if (best.Error < (prevBest.Error - m_noImprovementThreshold))
                    {
                        iterationsWithoutImprovement = 0;
                        prevBest = best;
                    }
                    else
                    {
                        iterationsWithoutImprovement++;
                    }

                    if (iterationsWithoutImprovement >= m_maxIterationsWithoutImprovement)
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
                    var refelctionScore = functionToMinimize(xr);

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
                        var expansionScore = functionToMinimize(xe);
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
                    var contractionScore = functionToMinimize(xc);
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
                        var score = functionToMinimize(xs);
                        nres.Add(score);
                    }

                    results = nres.ToList();
                }
            }

            return allResults.Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).ToArray();
        }

        /// <summary>
        /// Make sure the parameter set is within the specified bounds
        /// </summary>
        /// <param name="parameters"></param>
        void BoundCheck(double[] parameters)
        {
            for (int i = 0; i < parameters.Length; i++)
            {
                var range = m_parameters[i];
                parameters[i] = Math.Max(range.Min(), Math.Min(parameters[i], range.Max()));
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="newPoint"></param>
        void RandomRestartPoint(double[] newPoint)
        {
            // consider to implement gaussian selection of next point to avoid very similar point being selected
            // look at: https://github.com/ojdo/gbnm/blob/master/gbnm.m
            for (int i = 0; i < m_parameters.Length; i++)
            {
                var range = m_parameters[i];
                newPoint[i] = NewParameter(range.Min(), range.Max());
            }
        }

        /// <summary>
        /// Randomly select a parameter within the specified range
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        double NewParameter(double min, double max)
        {
            return m_random.NextDouble() * (max - min) + min;
        }
    }
}
