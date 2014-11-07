using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.Threading;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace SharpLearning.Linear.Optimization
{
    /// <summary>
    /// Gradient Descent optimization:
    /// http://en.wikipedia.org/wiki/Gradient_descent
    /// Works best with convex optimization objectives. If the function being minimized is not convex
    /// then there is a change the algorithm will get stuck in a local minima.
    /// </summary>
    public abstract class StochasticGradientDescent
    {
        protected readonly double m_learningRate;
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
            if (learningRate <= 0.0) { throw new ArgumentException("Learning rate must be larger than 0.0"); }
            if (iterations < 1) { throw new ArgumentException("Iterations must be at least 1"); }
            if (numberOfThreads < 1) { throw new ArgumentException("Number of threads must be at least 1"); }
            
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
            var observationsPrThread = targets.Length / m_numberOfThreads;
            var results = new ConcurrentBag<double[]>();
            var workers = new List<Action>();
            
            for (int i = 0; i < m_numberOfThreads; i++)
            {
                var interval = Interval1D.Create(0 + observationsPrThread * i,
                        observationsPrThread + (observationsPrThread * i));

                workers.Add(() => Iterate(observations, targets, new Random(m_random.Next()),
                    results, interval));
            }

            var m_threadedWorker = new WorkerRunner(workers);
            m_threadedWorker.Run();

            var models = results.ToArray();

            return AverageModels(observations.GetNumberOfColumns(), models);
        }

        /// <summary>
        /// Averages the parameters found for the models
        /// http://www.research.rutgers.edu/~lihong/pub/Zinkevich11Parallelized.pdf
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="models"></param>
        /// <returns></returns>
        double[] AverageModels(int numberOfFeatures, double[][] models)
        {
            var theta = new double[numberOfFeatures + 1];

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
            // initial theta + bias
            var theta = new double[x.GetNumberOfColumns() + 1];

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
        /// Abstract Gradient function.
        /// </summary>
        /// <param name="theta"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        protected abstract unsafe double[] Gradient(double[] theta, double* observation, double target);
    }
}
