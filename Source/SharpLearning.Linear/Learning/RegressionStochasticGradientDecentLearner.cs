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
        /// <param name="epochs">The number of parses over the data set (all obsrevations)</param>
        /// <param name="seed">Seed for the random number generator</param>
        /// <param name="numberOfThreads">Number of threads to use for paralization</param>
        public RegressionStochasticGradientDecentLearner(double learningRate, int epochs,
            int seed, int numberOfThreads)
        {
            m_stochasticGradientDescent =
                new SquareErrorStochasticGradientDescent(learningRate, epochs, seed, numberOfThreads);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="learningRate">The rate controls the step size at each gradient descent step. 
        /// A too small value can make the algorithms slow to converge and a too large values can make the algorithm not converge at all. 
        /// Meaning that the cost end of rising in each iteration</param>
        /// <param name="epochs">The number of parses over the data set (all obsrevations)</param>
        /// <param name="seed">Seed for the random number generator</param>
        public RegressionStochasticGradientDecentLearner(double learningRate = 0.001, int epochs = 5,
            int seed = 42)
            : this(learningRate, epochs, seed, System.Environment.ProcessorCount)
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

        /// <summary>
        /// Gradient Descent optimization for linear regression using square error loss:
        /// http://en.wikipedia.org/wiki/Gradient_descent
        /// Works best with convex optimization objectives. If the function being minimized is not convex
        /// then there is a change the algorithm will get stuck in a local minima.
        /// </summary>
        sealed class SquareErrorStochasticGradientDescent : StochasticGradientDescent
        {
            /// <summary>
            /// 
            /// </summary>
            /// <param name="learningRate">The rate controls the step size at each gradient descent step. 
            /// A too small value can make the algorithms slow to converge and a too large values can make the algorithm not converge at all. 
            /// Meaning that the cost end of rising in each iteration</param>
            /// <param name="epochs">The number of parses over the data set (all obsrevations)</param>
            /// <param name="seed">Seed for the random number generator</param>
            /// <param name="numberOfThreads">Number of threads to use for paralization</param>
            public SquareErrorStochasticGradientDescent (double learningRate, int epochs,
                int seed, int numberOfThreads)
                : base(learningRate, epochs, seed, numberOfThreads)
            {
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="learningRate">The rate controls the step size at each gradient descent step. 
            /// A too small value can make the algorithms slow to converge and a too large values can make the algorithm not converge at all. 
            /// Meaning that the cost end of rising in each iteration</param>
            /// <param name="epochs">The number of parses over the data set (all obsrevations)</param>
            /// <param name="seed">Seed for the random number generator</param>
            public SquareErrorStochasticGradientDescent(double learningRate = 0.001, int epochs = 5,
                int seed = 42)
                : base(learningRate, epochs, seed, System.Environment.ProcessorCount)
            {
            }


            /// <summary>
            /// Gradient function for linear regression objective.
            /// </summary>
            /// <param name="theta"></param>
            /// <param name="observations"></param>
            /// <param name="targets"></param>
            /// <returns></returns>
            protected override unsafe double[] Gradient(double[] theta, double* observation, double target)
            {
                // octave batch version
                // theta = theta - alpha * ((1/m) * ((X * theta) - y)' * X)';

                var error = (1 * theta[0]); // bias
                for (int i = 0; i < theta.Length - 1; i++)
                {
                    error += (observation[i] * theta[i + 1]);
                }

                error -= target;

                var regularization = 0.0; // 0.0 means no regularization
                theta[0] = theta[0] * (1.0 - m_learningRate * regularization) - 1 * error * m_learningRate; // bias

                for (int i = 0; i < theta.Length - 1; i++)
                {
                    theta[i + 1] = theta[i + 1] * (1.0 - m_learningRate * regularization) - observation[i] * error * m_learningRate;
                }

                return theta;
            }
        }
    }
}
