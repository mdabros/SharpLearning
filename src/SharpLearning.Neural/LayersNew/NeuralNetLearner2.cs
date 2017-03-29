using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.Optimizers;
using SharpLearning.Neural.TargetEncoders;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public class NeuralNetLearner2
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly NeuralNet2 Net;

        readonly int m_epochs;
        readonly int m_batchSize;

        readonly Random m_random;

        readonly NeuralNetOptimizer2 m_optimizer;
        readonly ILoss m_loss;

        /// <summary>
        /// Neural net learner. Controls the learning process using mini-batch gradient descent.
        /// </summary>
        /// <param name="net">The neural net to learn</param>
        /// <param name="loss">The loss measured and shown between each iteration</param>
        /// <param name="learningRate">Controls the step size when updating the weights. (Default is 0.001)</param>
        /// <param name="epochs">The maximum number of epochs before termination. (Default is 100)</param>
        /// <param name="batchSize">Batch size for mini-batch stochastic gradient descent. (Default is 128)</param>
        /// <param name="l1decay">L1 reguralization term. (Default is 0, so no reguralization)</param>
        /// <param name="l2decay">L2 reguralization term. (Default is 0, so no reguralization)</param>
        /// <param name="optimizerMethod">The method used for optimization (Default is RMSProp)</param>
        /// <param name="momentum">Momentum for gradient update. Should be between 0 and 1. (Defualt is 0.9)</param>
        /// <param name="rho">Squared gradient moving average decay factor (Default is 0.95)</param>
        /// <param name="beta1">Exponential decay rate for estimates of first moment vector, should be in range 0 to 1 (Default is 0.9)</param>
        /// <param name="beta2">Exponential decay rate for estimates of second moment vector, should be in range 0 to 1 (Default is 0.999)</param>
        public NeuralNetLearner2(NeuralNet2 net, ILoss loss, double learningRate = 0.001, int epochs = 100, int batchSize = 128, double l1decay = 0, double l2decay = 0,
            OptimizerMethod optimizerMethod = OptimizerMethod.RMSProp, double momentum = 0.9, double rho = 0.95, double beta1 = 0.9, double beta2 = 0.999)
        {
            if (net == null) { throw new ArgumentNullException("net"); }
            if (loss == null) { throw new ArgumentNullException("loss"); }
            if (learningRate <= 0) { throw new ArgumentNullException("learning rate must be larger than 0. Was: " + learningRate); }
            if (epochs <= 0) { throw new ArgumentNullException("Iterations must be larger than 0. Was: " + epochs); }
            if (batchSize <= 0) { throw new ArgumentNullException("batchSize must be larger than 0. Was: " + batchSize); }
            if (l1decay < 0) { throw new ArgumentNullException("l1decay must be positive. Was: " + l1decay); }
            if (l2decay < 0) { throw new ArgumentNullException("l1decay must be positive. Was: " + l2decay); }
            if (momentum <= 0) { throw new ArgumentNullException("momentum must be larger than 0. Was: " + momentum); }
            if (rho <= 0) { throw new ArgumentNullException("ro must be larger than 0. Was: " + rho); }
            if (beta1 <= 0) { throw new ArgumentNullException("beta1 must be larger than 0. Was: " + beta1); }
            if (beta2 <= 0) { throw new ArgumentNullException("beta2 must be larger than 0. Was: " + beta2); }

            Net = net;
            m_loss = loss;
            m_epochs = epochs;
            m_batchSize = batchSize;
            m_random = new Random(232);

            m_optimizer = new NeuralNetOptimizer2(learningRate, batchSize, l1decay, l2decay, 
                optimizerMethod, momentum, rho, beta1, beta2);
        }

        /// <summary>
        /// Learns a neural net based on the observations and targets.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public NeuralNet2 Learn(Tensor<double> observations, Tensor<double> targets)
        {
            var indices = Enumerable.Range(0, targets.Dimensions[0]).ToArray();
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
        public NeuralNet2 Learn(Tensor<double> observations, Tensor<double> targets, int[] indices)
        {          
            int numberOfTargets = NumberOfTargets(targets);
            var numberOfObservations = indices.Length;

            var inputShape = observations.Shape;
            var batchDimensions = new List<int> { m_batchSize };
            batchDimensions.AddRange(inputShape.Dimensions.Skip(1)); // skip

            var batchObservations = Tensor<double>.Build(batchDimensions.ToArray());
            var batchTargets = Tensor<double>.Build(m_batchSize, numberOfTargets);

            // hack because of missing input layer.
            var trainingInput = Variable.Create(batchObservations.Dimensions.ToArray());
            Net.Initialize(trainingInput, m_random);

            var parameters = new List<Data<double>>();
            Net.GetTrainableParameters(parameters);

            var lossFunc = new LogLoss();

            var batcher = new Batcher();
            batcher.Initialize(observations.Shape, 
                indices, m_random.Next());

            var timer = new Stopwatch();

            for (int epoch = 0; epoch < m_epochs; epoch++)
            {
                batcher.Shuffle();
                var accumulatedLoss = 0.0;

                timer.Restart();
                while (batcher.Next(m_batchSize,
                    Net, observations, targets,
                    batchObservations, batchTargets))
                {
                    Net.Forward();
                    Net.Backward();

                    var batchLoss = lossFunc.Loss(batchTargets, Net.BatchPredictions());
                    accumulatedLoss += batchLoss * m_batchSize;

                    m_optimizer.UpdateParameters(parameters);
                }
                timer.Stop();

                var loss = accumulatedLoss / (double)numberOfObservations;
                Trace.WriteLine($"Epoch: {epoch}, Loss: {loss}, Time (ms): {timer.ElapsedMilliseconds}");


                // validation
                //if (i % 5 == 0 && i != 0)
                //{
                //    for (int j = 0; j < validationObservations.Dimensions[0]; j++)
                //    {
                //        validationObservations.SliceCopy(j, 1, validationObservation);
                //        var prediction = m_net.Predict(validationObservation);
                //        validationPredictions.SetSlice(j, prediction);
                //    }

                //    var batchPredictionLoss = lossFunc.Loss(ValidationTargets, validationPredictions);
                //    Trace.WriteLine($"Validation Loss: {batchPredictionLoss}");

                //    // reset input to training.
                //    m_net.UpdateDimensions(trainingInput);
                //}
            }

            return Net.Copy();
        }

        static int NumberOfTargets(Tensor<double> targets)
        {
            // targets are assumed to be 1D or 2D            
            if (targets.Rank > 2 || targets.Rank == 0)
            {
                throw new ArgumentException($"Targets is expected to be 1D or 2D, was: {targets.Rank}");
            }

            var numberOfTargets = 0;
            if (targets.Rank == 1)
            {
                numberOfTargets = 1;
            }
            else
            {
                numberOfTargets = targets.Dimensions[1];
            }

            return numberOfTargets;
        }
    }
}
