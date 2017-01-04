using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.Optimizers;
using SharpLearning.Neural.TargetEncoders;
using System;
using System.Diagnostics;
using System.Linq;

namespace SharpLearning.Neural
{
    /// <summary>
    /// Neural net learner. Controls the learning process using mini-batch gradient descent.
    /// </summary>
    public class NeuralNetLearner
    {
        readonly NeuralNet m_net;

        readonly double m_learningRate;
        readonly double m_momentum;

        readonly int m_iterations;
        readonly int m_batchSize;

        readonly Random m_random;

        readonly NeuralNetOptimizer m_optimizer;
        readonly ITargetEncoder m_targetEncoder;
        readonly ILoss m_loss;

        int m_no_improvement_count;
        double m_best_loss;
        const double m_tol = 1e-4;

        /// <summary>
        /// Neural net learner. Controls the learning process using mini-batch gradient descent.
        /// </summary>
        /// <param name="net">The neural net to learn</param>
        /// <param name="targetEncoder">Controls how the training targets should be decoded. 
        /// This is different depending on if the net should be used for regression or classification.</param>
        /// <param name="loss">The loss measured and shown between each iteration</param>
        /// <param name="learningRate">Controls the step size when updating the weights. (Default is 0.01)</param>
        /// <param name="iterations">The maximum number of iterations before termination. (Default is 100)</param>
        /// <param name="batchSize">Batch size for mini-batch stochastic gradient descent. (Default is 128)</param>
        /// <param name="l1decay">L1 reguralization term. (Default is 0, so no reguralization)</param>
        /// <param name="l2decay">L2 reguralization term. (Default is 0, so no reguralization)</param>
        /// <param name="optimizerMethod">The method used for optimization (Default is Adagrad)</param>
        /// <param name="momentum">Momentum for gradient update. Should be between 0 and 1. (Defualt is 0.9)</param>
        /// <param name="rho">Squared gradient moving average decay factor (Default is 0.95)</param>
        /// <param name="beta1">Exponential decay rate for estimates of first moment vector, should be in range 0 to 1 (Default is 0.9)</param>
        /// <param name="beta2">Exponential decay rate for estimates of second moment vector, should be in range 0 to 1 (Default is 0.999)</param>
        public NeuralNetLearner(NeuralNet net, ITargetEncoder targetEncoder, ILoss loss, double learningRate = 0.01, int iterations = 100, int batchSize = 128, double l1decay = 0, double l2decay = 0,
            OptimizerMethod optimizerMethod = OptimizerMethod.Adagrad, double momentum = 0.9, double rho = 0.95, double beta1 = 0.9, double beta2 = 0.999)
        {
            if (net == null) { throw new ArgumentNullException("net"); }
            if (targetEncoder == null) { throw new ArgumentNullException("targetEncoder"); }
            if (loss == null) { throw new ArgumentNullException("loss"); }
            if (learningRate <= 0) { throw new ArgumentNullException("learning rate must be larger than 0. Was: " + learningRate); }
            if (iterations <= 0) { throw new ArgumentNullException("Iterations must be larger than 0. Was: " + iterations); }
            if (batchSize <= 0) { throw new ArgumentNullException("batchSize must be larger than 0. Was: " + batchSize); }
            if (l1decay < 0) { throw new ArgumentNullException("l1decay must be positive. Was: " + l1decay); }
            if (l2decay < 0) { throw new ArgumentNullException("l1decay must be positive. Was: " + l2decay); }
            if (momentum <= 0) { throw new ArgumentNullException("momentum must be larger than 0. Was: " + momentum); }
            if (rho <= 0) { throw new ArgumentNullException("ro must be larger than 0. Was: " + rho); }
            if (beta1 <= 0) { throw new ArgumentNullException("beta1 must be larger than 0. Was: " + beta1); }
            if (beta2 <= 0) { throw new ArgumentNullException("beta2 must be larger than 0. Was: " + beta2); }

            m_net = net;
            m_targetEncoder = targetEncoder;
            m_loss = loss;
            m_learningRate = learningRate;
            m_iterations = iterations;
            m_momentum = momentum;
            m_batchSize = batchSize;
            m_random = new Random(232);
            
            m_optimizer = new NeuralNetOptimizer(learningRate, batchSize, l1decay, l2decay, optimizerMethod, momentum, rho, beta1, beta2);

            SetupLinerAlgebraProvider();
        }

        /// <summary>
        /// Learns a neural net based on the observations and targets.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public NeuralNet Learn(F64Matrix observations, double[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Learns a neural net based on the observations and targets.
        /// The learning only uses the observations which indices are present in indices.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public NeuralNet Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            // targetEncoder 
            var oneOfNTargets = m_targetEncoder.Encode(targets);

            // Setup working parameters
            var samples = indices.Length;
            var learningIndices = indices.ToArray();
            var numberOfBatches = samples / m_batchSize; // check for size mismatch
            var batchTargets = Matrix<float>.Build.Dense(m_batchSize, oneOfNTargets.ColumnCount);
            var batchObservations = Matrix<float>.Build.Dense(m_batchSize, observations.GetNumberOfColumns());

            if (m_batchSize > samples)
            {
                throw new ArgumentException("BatchSize: " + m_batchSize +
                    " is larger than number og observations: " + samples);
            }

            var currentLoss = 0.0;
            m_best_loss = double.MaxValue;
            m_no_improvement_count = 0;

            // initialize net
            m_net.Initialize(m_batchSize, m_random);

            // extract reference to parameters and gradients
            var parametersAndGradients = m_net.GetParametersAndGradients();

            // reset optimizer
            m_optimizer.Reset();

            var timer = new Stopwatch();

            // train using stochastic gradient descent
            for (int iteration = 0; iteration < m_iterations; iteration++)
            {
                timer.Restart();

                var accumulatedLoss = 0.0;
                learningIndices.Shuffle(m_random);

                for (int i = 0; i < numberOfBatches; i++)
                {
                    var workIndices = learningIndices
                        .Skip(i * m_batchSize)
                        .Take(m_batchSize).ToArray();

                    if (workIndices.Length != m_batchSize)
                    {
                        continue; // only train with full batch size
                    }

                    CopyBatchTargets(oneOfNTargets, batchTargets, workIndices);
                    CopyBatch(observations, batchObservations, workIndices);

                    // forward pass.
                    var predictions = m_net.Forward(batchObservations);

                    // loss
                    var batchLoss = m_loss.Loss(batchTargets, predictions);
                    accumulatedLoss += batchLoss * m_batchSize;

                    // Backwards pass.
                    m_net.Backward(batchTargets);

                    // Weight update.
                    m_optimizer.UpdateParameters(parametersAndGradients);
                }

                timer.Stop();

                currentLoss = accumulatedLoss / (double)indices.Length;
                Trace.WriteLine("Iteration: " + (iteration + 1) + " Loss: " + currentLoss + " Time (ms): " + timer.ElapsedMilliseconds);

                UpdateNoImprovement(currentLoss);

                if (double.IsNaN(currentLoss))
                {
                    Trace.WriteLine("Loss is NaN, stopping...");
                    break;
                }
            }

            return m_net.CopyNetForPredictionModel();
        }

        void SetupLinerAlgebraProvider()
        {
            if (Control.TryUseNativeMKL())
            {
                Trace.WriteLine("Using MKL Provider");
            }
            else if (Control.TryUseNativeOpenBLAS())
            {
                Trace.WriteLine("Using OpenBLAS Provider");
            }
            else
            {
                Control.UseManaged();
                Control.UseMultiThreading();
                Trace.WriteLine("Using .Net Managed Provider");
            }
        }

        void UpdateNoImprovement(double currentLoss)
        {
            if (currentLoss > m_best_loss - m_tol)
            {
                m_no_improvement_count++;
            }
            else
            {
                m_no_improvement_count = 0;
            }

            if (currentLoss < m_best_loss)
            {
                m_best_loss = currentLoss;
            }
        }

        void CopyBatchTargets(Matrix<float> targets, Matrix<float> batch, int[] indices)
        {
            var cols = targets.ColumnCount;
            var batchRow = 0;
            foreach (var row in indices)
            {
                for (int col = 0; col < cols; col++)
                {
                    batch[batchRow, col] = targets[row, col];
                }
                batchRow++;
            }
        }

        void CopyBatch(F64Matrix observations, Matrix<float> batch, int[] indices)
        {
            var cols = observations.GetNumberOfColumns();
            var batchRow = 0;
            foreach (var row in indices)
            {
                for (int col = 0; col < cols; col++)
                {
                    batch[batchRow, col] = (float)observations[row, col];
                }
                batchRow++;
            }
        }
    }
}
