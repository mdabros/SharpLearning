using SharpLearning.Containers.Extensions;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SharpLearning.Neural.Optimizers
{


    /// <summary>
    /// Neural net optimizer for controlling the weight updates in neural net learning.
    /// uses mini-batch stochastic gradient descent. 
    /// Several different optimization methods is available through the constructor.
    /// </summary>
    public sealed class NeuralNetOptimizer
    {
        readonly float LearningRate;
        readonly float Momentum;
        readonly int BatchSize;

        readonly List<double[]> gsumWeights = new List<double[]>(); // last iteration gradients (used for momentum calculations)
        readonly List<double[]> xsumWeights = new List<double[]>(); // used in adam or adadelta

        readonly List<double[]> gsumBias = new List<double[]>(); // last iteration gradients (used for momentum calculations)
        readonly List<double[]> xsumBias = new List<double[]>(); // used in adam or adadelta


        int IterationCounter; // iteration counter

        OptimizerMethod OptimizerMethod = OptimizerMethod.Sgd;
        float Ro = 0.95f;
        float Eps = 1e-6f;
        float Beta1 = 0.9f;
        float Beta2 = 0.999f;

        float L1Decay = 0.0f;
        float L2Decay = 0.0f;

        /// <summary>
        /// Neural net optimizer for controlling the weight updates in neural net learning.
        /// uses mini-batch stochastic gradient descent. 
        /// Several different optimization methods is available through the constructor.
        /// </summary>
        /// <param name="learningRate">Controls the step size when updating the weights. (Default is 0.01)</param>
        /// <param name="batchSize">Batch size for mini-batch stochastic gradient descent. (Default is 128)</param>
        /// <param name="l1decay">L1 reguralization term. (Default is 0, so no reguralization)</param>
        /// <param name="l2decay">L2 reguralization term. (Default is 0, so no reguralization)</param>
        /// <param name="optimizerMethod">The method used for optimization (Default is Adagrad)</param>
        /// <param name="momentum">Momentum for gradient update. Should be between 0 and 1. (Defualt is 0.9)</param>
        /// <param name="ro"></param>
        /// <param name="beta1">Exponential decay rate for estimates of first moment vector, should be in range 0 to 1 (Default is 0.9)</param>
        /// <param name="beta2">Exponential decay rate for estimates of second moment vector, should be in range 0 to 1 (Default is 0.999)</param>
        public NeuralNetOptimizer(double learningRate, int batchSize, double l1decay=0, double l2decay=0, 
            OptimizerMethod optimizerMethod = OptimizerMethod.Adagrad, double momentum = 0.9, double ro=0.95, double beta1=0.9, double beta2=0.999)
        {
            if (learningRate <= 0) { throw new ArgumentNullException("learning rate must be larger than 0. Was: " + learningRate); }
            if (batchSize <= 0) { throw new ArgumentNullException("batchSize must be larger than 0. Was: " + batchSize); }
            if (l1decay < 0) { throw new ArgumentNullException("l1decay must be positive. Was: " + l1decay); }
            if (l2decay < 0) { throw new ArgumentNullException("l1decay must be positive. Was: " + l2decay); }
            if (momentum <= 0) { throw new ArgumentNullException("momentum must be larger than 0. Was: " + momentum); }
            if (ro <= 0) { throw new ArgumentNullException("ro must be larger than 0. Was: " + ro); }
            if (beta1 <= 0) { throw new ArgumentNullException("beta1 must be larger than 0. Was: " + beta1); }
            if (beta2 <= 0) { throw new ArgumentNullException("beta2 must be larger than 0. Was: " + beta2); }

            LearningRate = (float)learningRate;
            BatchSize = batchSize;
            L1Decay = (float)l1decay;
            L2Decay = (float)l2decay;
            
            OptimizerMethod = optimizerMethod;
            Momentum = (float)momentum;
            Ro = (float)ro;
            Beta1 = (float)beta1;
            Beta2 = (float)beta2;
        }

        /// <summary>
        /// Updates the parameters based on stochastic gradient descent.
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void UpdateParameters(List<ParametersAndGradients> parametersAndGradients)
        {
            this.IterationCounter++;
            // initialize lists for accumulators. Will only be done once on first iteration
            if (this.gsumWeights.Count == 0 && (this.OptimizerMethod != OptimizerMethod.Sgd || this.Momentum > 0.0))
            {
                // only vanilla sgd doesnt need either lists
                // momentum needs gsum
                // adagrad needs gsum
                // adam and adadelta needs gsum and xsum
                for (var i = 0; i < parametersAndGradients.Count; i++)
                {
                    this.gsumWeights.Add(new double[parametersAndGradients[i].Parameters.Weights.Data().Length]);
                    this.gsumBias.Add(new double[parametersAndGradients[i].Parameters.Bias.Data().Length]);
                    if (this.OptimizerMethod == OptimizerMethod.Adam || this.OptimizerMethod == OptimizerMethod.Adadelta)
                    {
                        this.xsumWeights.Add(new double[parametersAndGradients[i].Parameters.Weights.Data().Length]);
                        this.xsumBias.Add(new double[parametersAndGradients[i].Parameters.Bias.Data().Length]);
                    }
                }
            }

            // perform an update for all sets of weights
            Parallel.For(0, parametersAndGradients.Count, i => 
            {
                var parametersAndGradient = parametersAndGradients[i];
                // param, gradient, other options in future (custom learning rate etc)
                var parameters = parametersAndGradient.Parameters.Weights.Data();
                var parametersBias = parametersAndGradient.Parameters.Bias.Data();
                var gradients = parametersAndGradient.Gradients.Weights.Data();
                var gradientsBias = parametersAndGradient.Gradients.Bias.Data();

                // learning rate for some parameters.
                var l2DecayMul = 1.0;
                var l1DecayMul = 1.0;
                var l2Decay = this.L2Decay * l2DecayMul;
                var l1Decay = this.L1Decay * l1DecayMul;

                // update weights
                UpdateParam(i, parameters, gradients, l2Decay, l1Decay, gsumWeights, xsumWeights);
                    
                // Update biases
                UpdateParam(i, parametersBias, gradientsBias, l2Decay, l1Decay, gsumBias, xsumBias);
            });
        }

        private void UpdateParam(int i, float[] parameters, float[] gradients, double l2Decay, double l1Decay,
            List<double[]> gsum, List<double[]> xsum)
        {
            var plen = parameters.Length;
            for (var j = 0; j < plen; j++)
            {
                var l1Grad = l1Decay * (parameters[j] > 0 ? 1 : -1);
                var l2Grad = l2Decay * (parameters[j]);

                var gij = (l2Grad + l1Grad + gradients[j]) / this.BatchSize; // raw batch gradient

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

                switch (this.OptimizerMethod)
                {
                    case OptimizerMethod.Sgd:
                        {
                            if (this.Momentum > 0.0)
                            {
                                // momentum update
                                var dx = this.Momentum * gsumi[j] - this.LearningRate * gij; // step
                                gsumi[j] = dx; // back this up for next iteration of momentum
                                parameters[j] += (float)dx; // apply corrected gradient
                            }
                            else
                            {
                                // vanilla sgd
                                parameters[j] += (float)(-this.LearningRate * gij);
                            }
                        }
                        break;
                    case OptimizerMethod.Adam:
                        {
                            // adam update
                            gsumi[j] = gsumi[j] * this.Beta1 + (1 - this.Beta1) * gij; // update biased first moment estimate
                            xsumi[j] = xsumi[j] * this.Beta2 + (1 - this.Beta2) * gij * gij; // update biased second moment estimate
                            var biasCorr1 = gsumi[j] * (1 - Math.Pow(this.Beta1, this.IterationCounter)); // correct bias first moment estimate
                            var biasCorr2 = xsumi[j] * (1 - Math.Pow(this.Beta2, this.IterationCounter)); // correct bias second moment estimate
                            var dx = -this.LearningRate * biasCorr1 / (Math.Sqrt(biasCorr2) + this.Eps);
                            parameters[j] += (float)dx;
                        }
                        break;
                    case OptimizerMethod.Adagrad:
                        {
                            // adagrad update
                            gsumi[j] = gsumi[j] + gij * gij;
                            var dx = -this.LearningRate / Math.Sqrt(gsumi[j] + this.Eps) * gij;
                            parameters[j] += (float)dx;
                        }
                        break;
                    case OptimizerMethod.Adadelta:
                        {
                            // assume adadelta if not sgd or adagrad
                            gsumi[j] = this.Ro * gsumi[j] + (1 - this.Ro) * gij * gij;
                            var dx = -Math.Sqrt((xsumi[j] + this.Eps) / (gsumi[j] + this.Eps)) * gij;
                            xsumi[j] = this.Ro * xsumi[j] + (1 - this.Ro) * dx * dx; // yes, xsum lags behind gsum by 1.
                            parameters[j] += (float)dx;
                        }
                        break;
                    case OptimizerMethod.Windowgrad:
                        {
                            // this is adagrad but with a moving window weighted average
                            // so the gradient is not accumulated over the entire history of the run. 
                            // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
                            gsumi[j] = this.Ro * gsumi[j] + (1 - this.Ro) * gij * gij;
                            var dx = -this.LearningRate / Math.Sqrt(gsumi[j] + this.Eps) * gij;
                            // eps added for better conditioning
                            parameters[j] += (float)dx;
                        }
                        break;
                    case OptimizerMethod.Netsterov:
                        {
                            var dx = gsumi[j];
                            gsumi[j] = gsumi[j] * this.Momentum + this.LearningRate * gij;
                            dx = this.Momentum * dx - (1.0 + this.Momentum) * gsumi[j];
                            parameters[j] += (float)dx;
                        }
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }

                gradients[j] = 0.0f; // zero out gradient so that we can begin accumulating anew
            }
        }

        /// <summary>
        /// Resets the counters and momentum sums.
        /// </summary>
        public void Reset()
        {
            // clear counter
            this.IterationCounter = 0;
            
            // clear sums
            for (int i = 0; i < gsumWeights.Count; i++)
            {
                gsumWeights[i].Clear();
                gsumBias[i].Clear();
                xsumWeights[i].Clear();
                xsumBias[i].Clear();
            }
        }
    }
}
