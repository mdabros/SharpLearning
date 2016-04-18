using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpLearning.Containers.Arithmetic;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Optimization using the Nelder-Mead method also called downhill simplex method
    /// https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
    ///  It is applied to nonlinear optimization problems for which derivatives may not be known. 
    ///  However, the Nelder–Mead technique is a heuristic search method that can converge to non-stationary points.
    /// </summary>
    public sealed class NelderMeadOptimizer : IOptimizer
    {
        readonly double m_step;
        readonly double m_noImprovementThreshold;
        readonly int m_maxIterationsWithoutImprovement; 
        readonly int m_maxIteration;
        readonly double m_alpha;
        readonly double m_gamma;
        readonly double m_rho;
        readonly double m_sigma;
        readonly double[] m_initialParameters;

        /// <summary>
        /// Optimization using the Nelder-Mead method also called downhill simplex method
        /// https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
        ///  It is applied to nonlinear optimization problems for which derivatives may not be known. 
        ///  However, the Nelder–Mead technique is a heuristic search method that can converge to non-stationary points.
        /// </summary>
        /// <param name="initialParameters">Initial parameter set</param>
        /// <param name="step">Step size for each iteration (defualt is 0.1)</param>
        /// <param name="noImprovementThreshold">Minimum value of improvement before the improvement is accepted as an actual improvement (default is 10e-6)</param>
        /// <param name="maxIterationsWithoutImprovement">Maximum number of iterations without an improvement (default is 10)</param>
        /// <param name="max_iter">Maximum number of iterations. 0 is no limit and will run to convergens (default is 0)</param>
        /// <param name="alpha">Coefficient for reflection part of the algorithm (default is 1)</param>
        /// <param name="gamma">Coefficient for expansion part of the algorithm (default is 2)</param>
        /// <param name="rho">Coefficient for contraction part of the algorithm (default is -0.5)</param>
        /// <param name="sigma">Coefficient for shrink part of the algorithm (default is 0.5)</param>
        public NelderMeadOptimizer(double[] initialParameters, double step = 0.1, double noImprovementThreshold = 10e-6, int maxIterationsWithoutImprovement = 10, 
                int maxIteration = 0, double alpha = 1, double gamma = 2, double rho = -0.5, double sigma = 0.5)
        {
            if (initialParameters == null) { throw new ArgumentNullException("x_start"); }
            if (maxIterationsWithoutImprovement <= 0) { throw new ArgumentNullException("maxIterationsWithoutImprovement must be at least 1"); }

            m_step = step;
            m_noImprovementThreshold = noImprovementThreshold;
            m_maxIterationsWithoutImprovement = maxIterationsWithoutImprovement;
            m_maxIteration = maxIteration;
            m_alpha = alpha;
            m_gamma = gamma;
            m_rho = rho;
            m_sigma = sigma;
            m_initialParameters = initialParameters;
        }

        /// <summary>
        /// Optimization using the Nelder-Mead method also called downhill simplex method.
        /// Returns the result which best minimises the provided function.
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult OptimizeBest(Func<double[], OptimizerResult> functionToMinimize)
        {
            return Optimize(functionToMinimize).Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).First();
        }

        /// <summary>
        /// Optimization using the Nelder-Mead method also called downhill simplex method.
        /// Returns the final results ordered from best to worst (minimized).
        /// </summary>
        /// <param name="functionToMinimize"></param>
        /// <returns></returns>
        public OptimizerResult[] Optimize(Func<double[], OptimizerResult> functionToMinimize)
        {
            var dim = m_initialParameters.Length;
            var prevBest = functionToMinimize(m_initialParameters);
            var iterationsWithoutImprovement = 0;
            var results = new List<OptimizerResult> { new OptimizerResult(prevBest.ParameterSet, prevBest.Error) };

            for (int i = 0; i < dim; i++)
            {
                var x = m_initialParameters.ToArray();
                x[i] = x[i] + m_step;
                var score = functionToMinimize(x);
                results.Add(score);
            }

            // simplex iter
            var iterations = 0;
            while (iterations < m_maxIteration || m_maxIteration == 0)
            {
                results = results.OrderBy(r => r.Error).ToList();
                var best = results.First();

                // break after max_iter
                if (iterations >= m_maxIteration && m_maxIteration != 0)
                {
                    return results.ToArray();
                }

                iterations++;

                // break after no_improv_break iterations with no improvement
                //Trace.WriteLine("Current best:" + best.Error);

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
                    return results.ToArray();
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

                // reflection
                var last = results.Last();
                var xr = x0.Add(x0.Subtract(last.ParameterSet).Multiply(m_alpha));
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
                    var score = functionToMinimize(xs);
                    nres.Add(score);
                }

                results = nres.ToList();
            }

            return results.ToArray();
        }
    }
}
