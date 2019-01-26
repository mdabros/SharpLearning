using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization
{
    public delegate OptimizerResult HyperbandObjectiveFunction(double[] parameterSet, double budgetFactor);

    /// <summary>
    /// Hyperband optimizer based on:
    /// https://arxiv.org/pdf/1603.06560.pdf
    /// 
    /// Implementation based on:
    /// https://github.com/zygmuntz/hyperband
    /// 
    /// </summary>
    public sealed class HyperbandOptimizer
    {
        readonly IParameterSpec[] m_parameters;
        readonly IParameterSampler m_sampler;

        readonly int m_max_iter; // maximum number of iterations.
        readonly int m_eta; // defines configuration downsampling rate (default = 3)

        readonly int m_sMax;
        readonly int m_b;

        /// <summary>
        /// TODO: Find better names for arguments.
        /// </summary>
        /// <param name="maximunIterationsPrConfiguration"></param>
        /// <param name=""></param>
        public HyperbandOptimizer(IParameterSpec[] parameters, int maximunIterationsPrConfiguration = 81, int eta = 3)
        {
            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            m_sampler = new RandomUniform(seed: 34);
            // TODO: Add parameter checks.

            m_max_iter = maximunIterationsPrConfiguration;
            m_eta = eta;

            m_sMax =  (int)(Math.Log(m_max_iter) / Math.Log(m_eta));
            m_b = (m_sMax + 1) * m_max_iter;
        }

        public OptimizerResult[] Optimize(HyperbandObjectiveFunction functionToMinimize)
        {
            // Initialize the search
            var allResults = new List<OptimizerResult>();

            //for (int s = m_sMax + 1; s > 0; s--)
            for (int s = m_sMax; s >= 0; s--)
            {
                // initial number of configurations
                var n = (int)Math.Ceiling((m_b / m_max_iter) * (Math.Pow(m_eta, s) / (s + 1)));

                // initial number of iterations per config
                var r = m_max_iter * Math.Pow(m_eta, -s);

                // n random configurations
                var T = CreateParameterSets(m_parameters, n);
                var results = new ConcurrentBag<OptimizerResult>();
                for (int i = 0; i < s + 1; i++)
                {
                    // Run each of the n configs for <iterations> 
                    // and keep best (n_configs / eta) configurations

                    var n_configs = n * Math.Pow(m_eta, -i);
                    var n_iterations = r * Math.Pow(m_eta, i);

                    Trace.WriteLine($"{(int)Math.Round(n_configs)} configurations x {n_iterations:F1} iterations each");

                    var rangePartitioner = Partitioner.Create(T, true);
                    var options = new ParallelOptions { MaxDegreeOfParallelism = 8 };

                    Parallel.ForEach(rangePartitioner, options, (t, loopState) =>
                    {
                        var result = functionToMinimize(t, n_iterations);
                        results.Add(result);
                    });

                    // Select a number of best configurations for the next loop
                    T = results.OrderBy(v => v.Error)
                        .Take((int)Math.Round(n_configs / m_eta))
                        .Select(v => v.ParameterSet)
                        .ToArray();
                }
                allResults.AddRange(results);

                Trace.WriteLine($" Lowest loss so far: {allResults.OrderBy(v => v.Error).First().Error:F4}");
            }

            return allResults.ToArray();
        }

        double[][] CreateParameterSets(IParameterSpec[] parameters, 
            int setCount)
        {
            var newSearchSpace = new double[setCount][];
            for (int i = 0; i < newSearchSpace.Length; i++)
            {
                var newParameters = new double[parameters.Length];
                var index = 0;
                foreach (var param in parameters)
                {
                    newParameters[index] = param.SampleValue(m_sampler);
                    index++;
                }
                newSearchSpace[i] = newParameters;
            }

            return newSearchSpace;
        }
    }
}
