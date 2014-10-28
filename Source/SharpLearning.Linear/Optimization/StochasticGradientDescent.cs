using SharpLearning.Containers.Arithmetic;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers;
using System;
using System.Linq;
using System.Collections.Concurrent;
using System.Collections.Generic;
using SharpLearning.Threading;
using SharpLearning.Containers.Views;

namespace SharpLearning.Linear.Optimization
{
    /// <summary>
    /// Gradient Descent optimization:
    /// http://en.wikipedia.org/wiki/Gradient_descent
    /// Works best with convex optimization objectives. If the function being minimized is not convex
    /// then there is a change the algorithm will get stuck in a local minima.
    /// </summary>
    public sealed class StochasticGradientDescent
    {
        readonly double m_learningRate;
        readonly int m_iterations;
        readonly int m_numberOfThreads;
        readonly Random m_random;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="learningRate">The rate controls the step size at each gradient descent step. 
        /// A too small value can make the algorithms slow to converge and a too large values can make the algorithm not converge at all. 
        /// Meaning that the cost end of rising in each iteration</param>
        /// <param name="iterations">The number of gradient iterations</param>
        /// <param name="seed">Seed for the random number generator</param>
        /// <param name="numberOfThreads">Number of threads to use for paralization</param>
        public StochasticGradientDescent(double learningRate, int iterations,
            int seed, int numberOfThreads)
        {
            // add constructor checks
            m_learningRate = learningRate;
            m_iterations = iterations;
            m_numberOfThreads = numberOfThreads;
            m_random = new Random(seed);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="learningRate">The rate controls the step size at each gradient descent step. 
        /// A too small value can make the algorithms slow to converge and a too large values can make the algorithm not converge at all. 
        /// Meaning that the cost end of rising in each iteration</param>
        /// <param name="iterations">The number of gradient iterations</param>
        /// <param name="seed">Seed for the random number generator</param>
        public StochasticGradientDescent(double learningRate = 0.001, int iterations = 10000,
            int seed = 42)
            : this(learningRate, iterations, seed, System.Environment.ProcessorCount)
        {
        }

        /// <summary>
        /// Minimizes the target function using gradint descent. 
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public double[] Optimize(F64Matrix observations, double[] targets)
        {
            var bias = Enumerable.Range(0, targets.Length)
                .Select(v => 1.0).ToArray();

            var x = bias.CombineCols(observations);

            var m_numberOfThreads = 4;

            var observationsPrThread = targets.Length / m_numberOfThreads;
            var results = new ConcurrentBag<double[]>();
            var workers = new List<Action>();
            
            for (int i = 0; i < m_numberOfThreads; i++)
            {
                var interval = Interval1D.Create(0 + observationsPrThread * i,
                        observationsPrThread + (observationsPrThread * i));

                workers.Add(() => Iterate(x, targets, new Random(m_random.Next()),
                    results, interval));
            }

            var m_threadedWorker = new WorkerRunner(workers);
            m_threadedWorker.Run();

            var models = results.ToArray();

            return AverageModels(observations, models);
        }

        /// <summary>
        /// Averages the parameters found for the models
        /// http://www.research.rutgers.edu/~lihong/pub/Zinkevich11Parallelized.pdf
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="models"></param>
        /// <returns></returns>
        double[] AverageModels(F64Matrix observations, double[][] models)
        {
            var theta = new double[observations.GetNumberOfColumns() + 1];

            foreach (var model in models)
            {
                for (int i = 0; i < model.Length; i++)
                {
                    theta[i] += model[i];
                }
            }

            for (int i = 0; i < theta.Length; i++)
            {
                theta[i] = theta[i] / (double)models.Length;
            }

            return theta;
        }

        /// <summary>
        /// Runs local thread iterations
        /// </summary>
        /// <param name="x"></param>
        /// <param name="targets"></param>
        /// <param name="random"></param>
        /// <param name="models"></param>
        /// <param name="indices"></param>
        unsafe void Iterate(F64Matrix x, double[] targets, Random random,
            ConcurrentBag<double[]> models, Interval1D interval)
        {
            var theta = new double[x.GetNumberOfColumns()];

            using (var pinned = x.GetPinnedPointer())
            {
                var view = pinned.View();
                for (int i = 0; i < m_iterations; i++)
                {
                    var index = random.Next(interval.FromInclusive, interval.ToExclusive);
                    theta = Gradient(theta, view[index], targets[index]);
                }
            }

            models.Add(theta);
        }


        /// <summary>
        /// Temp gradient function for linear regression objective.
        /// </summary>
        /// <param name="theta"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        unsafe double[] Gradient(double[] theta, double* observation, double target)
        {
            // octave batch version
            // theta = theta - alpha * ((1/m) * ((X * theta) - y)' * X)';

            var error = 0.0;
            for (int i = 0; i < theta.Length; i++)
            {
                error += (observation[i] * theta[i]);
            }

            error -= target;
            
            for (int i = 0; i < theta.Length; i++)
            {
                var regularization = 0.0; // 0.0 means no regularization
                theta[i] = theta[i] * (1.0 - m_learningRate * regularization) - observation[i] * error * m_learningRate;
            }

            return theta;
        }
    }
}
