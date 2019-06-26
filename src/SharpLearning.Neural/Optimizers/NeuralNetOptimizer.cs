using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using SharpLearning.Containers.Extensions;

namespace SharpLearning.Neural.Optimizers
{
    /// <summary>
    /// Neural net optimizer for controlling the weight updates in neural net learning.
    /// uses mini-batch stochastic gradient descent. 
    /// Several different optimization methods is available through the constructor.
    /// </summary>
    public sealed class NeuralNetOptimizer
    {
        float m_learningRate;
        readonly float m_learningRateInit;
        readonly float m_momentum;
        readonly int m_batchSize;

        readonly List<double[]> m_gsumWeights = new List<double[]>(); // last iteration gradients (used for momentum calculations)
        readonly List<double[]> m_xsumWeights = new List<double[]>(); // used in adam or adadelta

        readonly OptimizerMethod m_optimizerMethod = OptimizerMethod.Sgd;
        readonly float m_rho = 0.95f;
        readonly float m_eps = 1e-8f;
        readonly float m_beta1 = 0.9f;
        readonly float m_beta2 = 0.999f;

        // Nadam specific members.
        readonly float m_schedule_decay = 0.004f;
        float m_schedule = 1.0f;
        double m_momentumCache;
        double m_momentumCache_1;
        double m_scheduleNew;
        double m_scheduleNext;

        readonly float m_l1Decay = 0.0f;
        readonly float m_l2Decay = 0.0f;

        int m_iterationCounter; // iteration counter

        /// <summary>
        /// Neural net optimizer for controlling the weight updates in neural net learning.
        /// uses mini-batch stochastic gradient descent. 
        /// Several different optimization methods is available through the constructor.
        /// </summary>
        /// <param name="learningRate">Controls the step size when updating the weights. (Default is 0.01)</param>
        /// <param name="batchSize">Batch size for mini-batch stochastic gradient descent. (Default is 128)</param>
        /// <param name="l1decay">L1 regularization term. (Default is 0, so no regularization)</param>
        /// <param name="l2decay">L2 regularization term. (Default is 0, so no regularization)</param>
        /// <param name="optimizerMethod">The method used for optimization (Default is RMSProp)</param>
        /// <param name="momentum">Momentum for gradient update. Should be between 0 and 1. (Default is 0.9)</param>
        /// <param name="rho">Squared gradient moving average decay factor (Default is 0.95)</param>
        /// <param name="beta1">Exponential decay rate for estimates of first moment vector, should be in range 0 to 1 (Default is 0.9)</param>
        /// <param name="beta2">Exponential decay rate for estimates of second moment vector, should be in range 0 to 1 (Default is 0.999)</param>
        public NeuralNetOptimizer(
            double learningRate, 
            int batchSize, 
            double l1decay = 0, 
            double l2decay = 0,
            OptimizerMethod optimizerMethod = OptimizerMethod.RMSProp, 
            double momentum = 0.9, 
            double rho = 0.95, 
            double beta1 = 0.9, 
            double beta2 = 0.999)
        {
            if (learningRate <= 0) { throw new ArgumentNullException("learning rate must be larger than 0. Was: " + learningRate); }
            if (batchSize <= 0) { throw new ArgumentNullException("batchSize must be larger than 0. Was: " + batchSize); }
            if (l1decay < 0) { throw new ArgumentNullException("l1decay must be positive. Was: " + l1decay); }
            if (l2decay < 0) { throw new ArgumentNullException("l1decay must be positive. Was: " + l2decay); }
            if (momentum <= 0) { throw new ArgumentNullException("momentum must be larger than 0. Was: " + momentum); }
            if (rho <= 0) { throw new ArgumentNullException("rho must be larger than 0. Was: " + rho); }
            if (beta1 <= 0) { throw new ArgumentNullException("beta1 must be larger than 0. Was: " + beta1); }
            if (beta2 <= 0) { throw new ArgumentNullException("beta2 must be larger than 0. Was: " + beta2); }

            m_learningRate = (float)learningRate;
            m_learningRateInit = (float)learningRate;
            m_batchSize = batchSize;
            m_l1Decay = (float)l1decay;
            m_l2Decay = (float)l2decay;

            m_optimizerMethod = optimizerMethod;
            m_momentum = (float)momentum;
            m_rho = (float)rho;
            m_beta1 = (float)beta1;
            m_beta2 = (float)beta2;
        }

        /// <summary>
        /// Updates the parameters based on stochastic gradient descent.
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void UpdateParameters(List<ParametersAndGradients> parametersAndGradients)
        {
            m_iterationCounter++;

            // initialize accumulators. Will only be done once on first iteration and if optimizer methods is not sgd
            var useAccumulators = m_gsumWeights.Count == 0 && 
                (m_optimizerMethod != OptimizerMethod.Sgd || m_momentum > 0.0);

            if (useAccumulators)
            {
                InitializeAccumulators(parametersAndGradients);
            }

            UpdateLearningRate();

            // perform update of all parameters
            Parallel.For(0, parametersAndGradients.Count, i =>
            {
                var parametersAndGradient = parametersAndGradients[i];

                // extract parameters and gradients
                var parameters = parametersAndGradient.Parameters;
                var gradients = parametersAndGradient.Gradients;

                // update weights
                UpdateParam(i, parameters, gradients, m_l2Decay, m_l1Decay, m_gsumWeights, m_xsumWeights);
            });
        }

        void InitializeAccumulators(List<ParametersAndGradients> parametersAndGradients)
        {
            for (var i = 0; i < parametersAndGradients.Count; i++)
            {
                m_gsumWeights.Add(new double[parametersAndGradients[i].Parameters.Length]);
                if (m_optimizerMethod == OptimizerMethod.Adam || 
                    m_optimizerMethod == OptimizerMethod.Adadelta || 
                    m_optimizerMethod == OptimizerMethod.AdaMax ||
                    m_optimizerMethod == OptimizerMethod.Nadam)
                {
                    m_xsumWeights.Add(new double[parametersAndGradients[i].Parameters.Length]);
                }
            }
        }

        void UpdateLearningRate()
        {
            switch (m_optimizerMethod)
            {
                case OptimizerMethod.Adam:
                    {
                        m_learningRate = (float)(m_learningRateInit * Math.Sqrt(1.0 - Math.Pow(m_beta2, m_iterationCounter)) /
                            (1 - Math.Pow(m_beta1, m_iterationCounter)));
                    }
                    break;
                case OptimizerMethod.AdaMax:
                    {
                        m_learningRate = (float)(m_learningRateInit / (1.0 - Math.Pow(m_beta1, m_iterationCounter)));
                    }
                    break;
                case OptimizerMethod.Nadam:
                    {
                        // Nadam does not update learning rate but updates the schedule for the momentum cache.
                        m_momentumCache = m_beta1 * (1.0 - 0.5 * (Math.Pow(0.96, m_iterationCounter * m_schedule_decay)));
                        m_momentumCache_1 = m_beta1 * (1.0 - 0.5 * (Math.Pow(0.96, (m_iterationCounter + 1) * m_schedule_decay)));

                        m_scheduleNew = m_schedule * m_momentumCache;
                        m_scheduleNext = m_schedule * m_momentumCache * m_momentumCache_1;
                        m_schedule = (float)m_scheduleNew;
                    }
                    break;
            }
        }

        void UpdateParam(int i, float[] parameters, float[] gradients, double l2Decay, double l1Decay,
            List<double[]> gsum, List<double[]> xsum)
        {
            for (var j = 0; j < parameters.Length; j++)
            {
                var l1Grad = l1Decay * (parameters[j] > 0 ? 1 : -1);
                var l2Grad = l2Decay * (parameters[j]);

                var gij = (l2Grad + l1Grad + gradients[j]) / m_batchSize; // raw batch gradient

                double[] gsumi = null;
                if (gsum.Count > 0)
                {
                    gsumi = gsum[i];
                }

                double[] xsumi = null;
                if (xsum.Count > 0)
                {
                    xsumi = xsum[i];
                }

                switch (m_optimizerMethod)
                {
                    case OptimizerMethod.Sgd:
                        {
                            if (m_momentum > 0.0) // sgd + momentum
                            {
                                var dx = m_momentum * gsumi[j] - m_learningRate * gij;
                                gsumi[j] = dx; 
                                parameters[j] += (float)dx; 
                            }
                            else // standard sgd
                            {
                                parameters[j] += (float)(-m_learningRate * gij);
                            }
                        }
                        break;
                    case OptimizerMethod.Adam:
                        {
                            gsumi[j] = m_beta1 * gsumi[j] + (1.0 - m_beta1) * gij; 
                            xsumi[j] = m_beta2 * xsumi[j] + (1.0 - m_beta2) * gij * gij;

                            var dx = -m_learningRate * gsumi[j] / (Math.Sqrt(xsumi[j]) + m_eps);
                            parameters[j] += (float)dx;
                        }
                        break;
                    case OptimizerMethod.AdaMax:
                        {
                            gsumi[j] = m_beta1 * gsumi[j] + (1.0 - m_beta1) * gij; 
                            xsumi[j] = Math.Max(m_beta2 * xsumi[j], Math.Abs(gij)); 

                            var dx = -m_learningRate * gsumi[j] / (xsumi[j] + m_eps);
                            parameters[j] += (float)dx;
                        }
                        break;
                    case OptimizerMethod.Nadam:
                        {
                            var gPrime = gij / (1.0 - m_scheduleNew);
                            gsumi[j] = m_beta1 * gsumi[j] + (1.0 - m_beta1) * gij;
                            var gsumiPrime = gsumi[j] / (1.0 - m_scheduleNext);

                            xsumi[j] = m_beta2 * xsumi[j] + (1.0 - m_beta2) * gij * gij;
                            var xsumiPrime = xsumi[j] / (1.0 - Math.Pow(m_beta2, m_iterationCounter));
                            var gsumiBar = (1.0 - m_momentumCache) * gPrime + m_momentumCache_1 * gsumiPrime;

                            var dx = -m_learningRate * gsumiBar / (Math.Sqrt(xsumiPrime) + m_eps);
                            parameters[j] += (float)dx;
                        }
                        break;
                    case OptimizerMethod.Adagrad:
                        {
                            gsumi[j] = gsumi[j] + gij * gij;
                            var dx = -m_learningRate * gij / Math.Sqrt(gsumi[j] + m_eps);
                            parameters[j] += (float)dx;
                        }
                        break;
                    case OptimizerMethod.RMSProp:
                        {
                            gsumi[j] = m_rho * gsumi[j] + (1.0 - m_rho) * gij * gij;
                            var dx = -m_learningRate * gij / (Math.Sqrt(gsumi[j]) + m_eps);
                            parameters[j] += (float)dx;
                        }
                        break;
                    case OptimizerMethod.Adadelta:
                        {
                            gsumi[j] = m_rho * gsumi[j] + (1 - m_rho) * gij * gij;
                            
                            // learning rate multiplication left out since recommended default is 1.0. 
                            var dx = - gij * Math.Sqrt(xsumi[j] + m_eps) / Math.Sqrt(gsumi[j] + m_eps); 
                            xsumi[j] = m_rho * xsumi[j] + (1 - m_rho) * dx * dx;

                            parameters[j] += (float)dx;
                        }
                        break;
                    case OptimizerMethod.Netsterov:
                        {
                            var dx = gsumi[j];
                            gsumi[j] = gsumi[j] * m_momentum + m_learningRate * gij;
                            dx = m_momentum * dx - (1.0 + m_momentum) * gsumi[j];
                            parameters[j] += (float)dx;
                        }
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }

                gradients[j] = 0.0f; // zero out gradient between each iteration
            }
        }

        /// <summary>
        /// Resets the counters and momentum sums.
        /// </summary>
        public void Reset()
        {
            // clear counter
            m_iterationCounter = 0;

            // clear sums
            for (int i = 0; i < m_gsumWeights.Count; i++)
            {
                m_gsumWeights[i].Clear();
            }

            for (int i = 0; i < m_xsumWeights.Count; i++)
            {
                m_xsumWeights[i].Clear();
            }
        }
    }
}
