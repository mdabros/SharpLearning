using SharpLearning.Containers.Arithmetic;
using SharpLearning.Containers.Matrices;
using System.Linq;

namespace SharpLearning.Linear.Optimization
{
    /// <summary>
    /// Gradient Descent optimization:
    /// http://en.wikipedia.org/wiki/Gradient_descent
    /// Works best with convex optimization objectives. If the function being minimized is not convex
    /// then there is a change the algorithm will get stuck in a local minima.
    /// </summary>
    public sealed class GradientDescent
    {
        readonly double m_learningRate;
        readonly int m_iterations;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="learningRate">The rate controls the step size at each gradient descent step. 
        /// A too small value can make the algorithms slow to converge and a too large values can make the algorithm not converge at all. 
        /// Meaning that the cost end of rising in each iteration</param>
        /// <param name="iterations">The number of gradient iterations</param>
        public GradientDescent(double learningRate=0.01, int iterations=400)
        {
            m_learningRate = learningRate;
            m_iterations = iterations;
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
            var theta = new double[observations.GetNumberOfColumns() + 1];

            for (int i = 0; i < m_iterations; i++)
            {
                theta = Gradient(theta, x, targets);
                //Trace.WriteLine("Cost: " + Cost(theta, x, targets));
            }

            return theta;
        }

        /// <summary>
        /// Temp gradient function for linear regression objective.
        /// </summary>
        /// <param name="theta"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        double[] Gradient(double[] theta, F64Matrix observations, double[] targets)
        {
            //theta = theta - alpha * ((1/m) * ((X * theta) - y)' * X)';
            
            var m = observations.GetNumberOfRows();
            var temp1 = (observations.Multiply(theta).Subtract(targets));
            var temp2 = observations.Transpose().Multiply(temp1);
            var update = theta.Subtract(temp2.Multiply(m_learningRate / m));

            return update;
        }

        /// <summary>
        /// Temp cost function for linear regression objective.
        /// </summary>
        /// <param name="theta"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        double Cost(double[] theta, F64Matrix observations, double[] targets)
        {

            //predictions = X * theta;
            //sqrError = (predictions - y).^2;

            //m = size(X, 1);
            //J = 1/(2*m) * sum(sqrError);

            var predictions = observations.Multiply(theta);
            var errorSum = 0.0;
            double m = observations.GetNumberOfRows();

            for (int i = 0; i < predictions.Length; i++)
            {
                var error = predictions[i]- targets[i];
                errorSum += (error * error);
            }

            var cost = 1.0 / (2.0 * m) * errorSum;
            return cost;
        }
    }
}
