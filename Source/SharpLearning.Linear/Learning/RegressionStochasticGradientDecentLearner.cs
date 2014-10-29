using SharpLearning.Containers.Matrices;
using SharpLearning.Linear.Models;
using SharpLearning.Linear.Optimization;

namespace SharpLearning.Linear.Learning
{
    /// <summary>
    /// Regression learner using stochastic gradient descent for optimizing the model. 
    /// Stochastic gradient descent operates best when all features are equally scaled. 
    /// For example between 0.0 and 1.0 
    /// </summary>
    public sealed class RegressionStochasticGradientDecentLearner
    {
        readonly StochasticGradientDescent m_stochasticGradientDescent;
        // Add loss functions (Huber, EN, squared), regularization parameter and so forth

        /// <summary>
        /// 
        /// </summary>
        /// <param name="learningRate">The rate controls the step size at each gradient descent step. 
        /// A too small value can make the algorithms slow to converge and a too large values can make the algorithm not converge at all. 
        /// Meaning that the cost end of rising in each iteration</param>
        /// <param name="iterations">The number of gradient iterations</param>
        /// <param name="seed">Seed for the random number generator</param>
        /// <param name="numberOfThreads">Number of threads to use for paralization</param>
        public RegressionStochasticGradientDecentLearner(double learningRate, int iterations,
            int seed, int numberOfThreads)
        {
            m_stochasticGradientDescent = 
                new StochasticGradientDescent(learningRate, iterations, seed, numberOfThreads);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="learningRate">The rate controls the step size at each gradient descent step. 
        /// A too small value can make the algorithms slow to converge and a too large values can make the algorithm not converge at all. 
        /// Meaning that the cost end of rising in each iteration</param>
        /// <param name="iterations">The number of gradient iterations</param>
        /// <param name="seed">Seed for the random number generator</param>
        public RegressionStochasticGradientDecentLearner(double learningRate = 0.001, int iterations = 10000,
            int seed = 42)
            : this(learningRate, iterations, seed, System.Environment.ProcessorCount)
        {
        }

        
        /// <summary>
        /// Learns a linear regression model using StochasticGradientDecent
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public RegressionStochasticGradientDecentModel Learn(F64Matrix observations, double[] targets)
        {
            var weights = m_stochasticGradientDescent.Optimize(observations, targets);
            return new RegressionStochasticGradientDecentModel(weights);
        }
    }
}
