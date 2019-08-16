using System;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.Models;
using SharpLearning.Neural.Optimizers;
using SharpLearning.Neural.TargetEncoders;

namespace SharpLearning.Neural.Learners
{
    /// <summary>
    /// ClassificationNeuralNet learner using mini-batch gradient descent. 
    /// Several optimization methods is available through the constructor.
    /// </summary>
    public sealed class ClassificationNeuralNetLearner 
        : IIndexedLearner<double>
        , IIndexedLearner<ProbabilityPrediction>
        , ILearner<double>
        , ILearner<ProbabilityPrediction>
    {
        readonly NeuralNetLearner m_learner;

        /// <summary>
        /// ClassificationNeuralNet learner using mini-batch gradient descent. 
        /// Several optimization methods is available through the constructor.
        /// </summary>
        /// <param name="net">The neural net to learn</param>
        /// <param name="loss">The loss measured and shown between each iteration</param>
        /// <param name="learningRate">Controls the step size when updating the weights. (Default is 0.001)</param>
        /// <param name="iterations">The maximum number of iterations before termination. (Default is 100)</param>
        /// <param name="batchSize">Batch size for mini-batch stochastic gradient descent. (Default is 128)</param>
        /// <param name="l1decay">L1 regularization term. (Default is 0, so no regularization)</param>
        /// <param name="l2decay">L2 regularization term. (Default is 0, so no regularization)</param>
        /// <param name="optimizerMethod">The method used for optimization (Default is RMSProp)</param>
        /// <param name="momentum">Momentum for gradient update. Should be between 0 and 1. (Default is 0.9)</param>
        /// <param name="rho">Squared gradient moving average decay factor (Default is 0.95)</param>
        /// <param name="beta1">Exponential decay rate for estimates of first moment vector, should be in range 0 to 1 (Default is 0.9)</param>
        /// <param name="beta2">Exponential decay rate for estimates of second moment vector, should be in range 0 to 1 (Default is 0.999)</param>
        public ClassificationNeuralNetLearner(NeuralNet net, 
            ILoss loss, 
            double learningRate = 0.001, 
            int iterations = 100, 
            int batchSize = 128, 
            double l1decay = 0, double l2decay = 0,
            OptimizerMethod optimizerMethod = OptimizerMethod.RMSProp, 
            double momentum = 0.9, 
            double rho = 0.95, 
            double beta1 = 0.9, 
            double beta2 = 0.999)
        {
            if (!(net.Layers.Last() is IClassificationLayer))
            {
                throw new ArgumentException("Last layer must be a classification layer type. Was: " 
                    + net.Layers.Last().GetType().Name);
            }

            m_learner = new NeuralNetLearner(net, new OneOfNTargetEncoder(), loss,
                learningRate, iterations, batchSize, l1decay, l2decay, 
                optimizerMethod, momentum, rho, beta1, beta2);
        }

        /// <summary>
        /// Learns a classification neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public ClassificationNeuralNetModel Learn(F64Matrix observations, double[] targets)
        {
            var targetNames = GetOrderedTargetNames(targets);
            var model = m_learner.Learn(observations, targets);
            return new ClassificationNeuralNetModel(model, targetNames);
        }

        /// <summary>
        /// Learns a classification neural network
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public ClassificationNeuralNetModel Learn(F64Matrix observations, double[] targets, 
            int[] indices)
        {
            var targetNames = GetOrderedTargetNames(targets);
            var model = m_learner.Learn(observations, targets, indices);
            return new ClassificationNeuralNetModel(model, targetNames);
        }

        /// <summary>
        /// Learns a classification neural network.
        /// ValidationObservations and ValidationTargets are used to track the validation loss pr. iteration.
        /// The iteration with the best validation loss is returned.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="validationObservations"></param>
        /// <param name="validationTargets"></param>
        /// <returns></returns>
        public ClassificationNeuralNetModel Learn(F64Matrix observations, double[] targets,
            F64Matrix validationObservations, double[] validationTargets)
        {
            var targetNames = GetOrderedTargetNames(targets);

            var model = m_learner.Learn(observations, targets, 
                validationObservations, validationTargets);

            return new ClassificationNeuralNetModel(model, targetNames);
        }

        /// <summary>
        /// Private explicit interface implementation for indexed learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictorModel<double> IIndexedLearner<double>.Learn(
            F64Matrix observations, double[] targets, int[] indices) => Learn(observations, targets, indices);

        /// <summary>
        /// Private explicit interface implementation for indexed probability learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictorModel<ProbabilityPrediction> IIndexedLearner<ProbabilityPrediction>.Learn(
            F64Matrix observations, double[] targets, int[] indices) => Learn(observations, targets, indices);

        /// <summary>
        /// Private explicit interface implementation.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<double> ILearner<double>.Learn(
            F64Matrix observations, double[] targets) => Learn(observations, targets);

        /// <summary>
        /// Private explicit interface implementation for probability learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<ProbabilityPrediction> ILearner<ProbabilityPrediction>.Learn(
            F64Matrix observations, double[] targets) => Learn(observations, targets);

        double[] GetOrderedTargetNames(double[] targets)
        {
            return targets.Distinct().OrderBy(v => v).ToArray();
        }
    }
}
