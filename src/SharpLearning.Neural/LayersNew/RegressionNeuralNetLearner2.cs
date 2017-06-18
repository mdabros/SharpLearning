using System.Collections.Generic;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.Optimizers;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// RegressionNeuralNet learner using mini-batch gradient descent. 
    /// Several optimization methods is availible through the constructor.
    /// </summary>
    public sealed class RegressionNeuralNetLearner2 : IIndexedLearner<double>, ILearner<double>
    {
        readonly NeuralNetLearner2 m_learner;

        /// <summary>
        /// RegressionNeuralNet learner using mini-batch gradient descent. 
        /// Several optimization methods is availible through the constructor.
        /// </summary>
        /// <param name="net">The neural net to learn</param>
        /// <param name="loss">The loss measured and shown between each iteration</param>
        /// <param name="learningRate">Controls the step size when updating the weights. (Default is 0.001)</param>
        /// <param name="iterations">The maximum number of iterations before termination. (Default is 100)</param>
        /// <param name="batchSize">Batch size for mini-batch stochastic gradient descent. (Default is 128)</param>
        /// <param name="l1decay">L1 reguralization term. (Default is 0, so no reguralization)</param>
        /// <param name="l2decay">L2 reguralization term. (Default is 0, so no reguralization)</param>
        /// <param name="optimizerMethod">The method used for optimization (Default is RMSProp)</param>
        /// <param name="momentum">Momentum for gradient update. Should be between 0 and 1. (Defualt is 0.9)</param>
        /// <param name="rho">Squared gradient moving average decay factor (Default is 0.95)</param>
        /// <param name="beta1">Exponential decay rate for estimates of first moment vector, should be in range 0 to 1 (Default is 0.9)</param>
        /// <param name="beta2">Exponential decay rate for estimates of second moment vector, should be in range 0 to 1 (Default is 0.999)</param>
        public RegressionNeuralNetLearner2(NeuralNet2 net, ILoss loss, double learningRate = 0.001, int iterations = 100, int batchSize = 128, double l1decay = 0, double l2decay = 0,
            OptimizerMethod optimizerMethod = OptimizerMethod.RMSProp, double momentum = 0.9, double rho = 0.95, double beta1 = 0.9, double beta2 = 0.999)
        {
            m_learner = new NeuralNetLearner2(net, loss, learningRate, iterations, batchSize, 
                l1decay, l2decay, optimizerMethod, momentum, rho, beta1, beta2);
        }

        /// <summary>
        /// Learns a Regression neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public RegressionNeuralNetModel2 Learn(F64Matrix observations, double[] targets)
        {
            var indices = Enumerable.Range(0, observations.RowCount).ToArray();
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Learns a Regression neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public RegressionNeuralNetModel2 Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            var tensorTargets = Tensor<double>.Build(targets, targets.Length, 1); // assume single target value for regression.

            var observationsShape = new List<int> { observations.RowCount };
            observationsShape.AddRange(m_learner.Net.Input.Dimensions.Skip(1).ToArray());

            var tensorObservations = Tensor<double>.Build(observations.Data(), observationsShape.ToArray());

            var model = m_learner.Learn(tensorObservations, tensorTargets, indices);
            return new RegressionNeuralNetModel2(model);
        }

        /// <summary>
        /// Learns a Regression neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        IPredictorModel<double> IIndexedLearner<double>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
        }
    }
}
