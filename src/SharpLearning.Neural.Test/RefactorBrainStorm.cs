using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Neural.Test.RefactorBranStorm
{
    [TestClass]
    public class RefactorBrainStormTest
    {
        [TestMethod]
        public void RefactorBrainStorm()
        {
            var inputShape = new TensorShape(10, 20);
            var targetShape = new TensorShape(1);

            var net = new NeuralNet();
            net.Add(new InputLayer(inputShape.Dimensions));
            net.Add(new DenseLayer(units: 128));
            net.Add(new ReluLayer());
            net.Add(new DenseLayer(units: 1));

            var loss = new MeanSquareLoss();

            var random = new Random(232);
            var storage = new NeuralNetStorage();

            var batchSize = 32;
            net.Initialize(random, batchSize, storage);
            Trace.WriteLine(net.Summary());

            var minibatchSource = new MinibatchSource(inputShape, targetShape, seed: random.Next());

            var iterations = 100;

            var optimizer = new SgdOptimizer(learningRate: 0.01f, batchSize: batchSize);

            for (int iteration = 0; iteration < iterations; iteration++)
            {
                var (observations, targets) = minibatchSource.GetMinibatch(batchSize);
                
                // inputs are assigned to the first layer.
                storage.AssignTensor(net.Input, observations);               

                // forward 
                net.Forward(storage);

                // calculate losses.
                var predictions = storage.GetTensor(net.Output);
                var sampleLosses = loss.SampleLosses(targets, predictions);
                var currentLoss = loss.AccumulateSampleLoss(sampleLosses);

                Trace.WriteLine("Batch Loss:" + currentLoss);

                // assign sample losses and back propagate.
                storage.AssignGradient(net.Output, sampleLosses);
                net.Backward(storage);

                // update parameters.
                var parameters = storage.GetTrainableParameters();
                optimizer.UpdateParameters(parameters);
            }
        }

        public class MinibatchSource
        {
            readonly TensorShape m_sampleShape;
            readonly TensorShape m_targetShape;
            readonly Random m_random;

            public MinibatchSource(TensorShape sampleShape, TensorShape targetShape,
                int seed)
            {
                m_sampleShape = sampleShape;
                m_targetShape = targetShape;
                m_random = new Random(seed);
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="batchSize"></param>
            /// <param name="net"></param>
            /// <param name="targets"></param>
            /// <param name="targets"></param>
            /// <param name="batchtargets"></param>
            /// <param name="BatchTargets"></param>
            public (Tensor<float> observations, Tensor<float> targets) GetMinibatch(int batchSize)
            {
                var observationsDimensions = new List<int> { batchSize };
                observationsDimensions.AddRange(m_sampleShape.Dimensions);
                var observations = new Tensor<float>(observationsDimensions.ToArray(), DataLayout.RowMajor);
                observations.Map(() => (float)m_random.NextDouble());

                var targetsDimensions = new List<int> { batchSize };
                targetsDimensions.AddRange(m_targetShape.Dimensions);
                var targets = new Tensor<float>(targetsDimensions.ToArray(), DataLayout.RowMajor);
                targets.Map(() => (float)m_random.NextDouble());

                return (observations, targets);
            }
        }

        /// <summary>
        /// Neural net optimizer for controlling the weight updates in neural net learning.
        /// uses mini-batch stochastic gradient descent. 
        /// Several different optimization methods is available through the constructor.
        /// </summary>
        public sealed class SgdOptimizer
        {
            float m_learningRate;
            readonly int m_batchSize;

            readonly float m_l1Decay = 0.0f;
            readonly float m_l2Decay = 0.0f;

            /// <summary>
            /// Neural net optimizer for controlling the weight updates in neural net learning.
            /// uses mini-batch stochastic gradient descent. 
            /// Several different optimization methods is available through the constructor.
            /// </summary>
            /// <param name="learningRate">Controls the step size when updating the weights. (Default is 0.01)</param>
            /// <param name="batchSize">Batch size for mini-batch stochastic gradient descent. (Default is 128)</param>
            /// <param name="l1decay">L1 regularization term. (Default is 0, so no regularization)</param>
            /// <param name="l2decay">L2 regularization term. (Default is 0, so no regularization)</param>
            public SgdOptimizer(float learningRate, int batchSize, float l1decay = 0, float l2decay = 0)
            {
                if (learningRate <= 0) { throw new ArgumentNullException("learning rate must be larger than 0. Was: " + learningRate); }
                if (batchSize <= 0) { throw new ArgumentNullException("batchSize must be larger than 0. Was: " + batchSize); }
                if (l1decay < 0) { throw new ArgumentNullException("l1decay must be positive. Was: " + l1decay); }
                if (l2decay < 0) { throw new ArgumentNullException("l1decay must be positive. Was: " + l2decay); }

                m_learningRate = learningRate;
                m_batchSize = batchSize;
                m_l1Decay = l1decay;
                m_l2Decay = l2decay;
            }

            /// <summary>
            /// Updates the parameters based on stochastic gradient descent.
            /// </summary>
            /// <param name="parametersAndGradients"></param>
            public void UpdateParameters(List<Data<float>> parametersAndGradients)
            {
                // perform update of all parameters
                Parallel.For(0, parametersAndGradients.Count, i =>
                {
                    var parametersAndGradient = parametersAndGradients[i];

                    // extract parameters and gradients
                    var parameters = parametersAndGradient.Tensor.Data;
                    var gradients = parametersAndGradient.Gradient.Data;

                    // update weights
                    UpdateParam(i, parameters, gradients, m_l2Decay, m_l1Decay);
                });
            }

            void UpdateParam(int i, float[] parameters, float[] gradients, float l2Decay, float l1Decay)
            {
                for (var j = 0; j < parameters.Length; j++)
                {
                    var l1Grad = l1Decay * (parameters[j] > 0 ? 1 : -1);
                    var l2Grad = l2Decay * (parameters[j]);

                    var gradient = (l2Grad + l1Grad + gradients[j]) / m_batchSize; // batch gradient

                    // Standard sgd
                    parameters[j] += (-m_learningRate * gradient);
                    gradients[j] = 0.0f; // zero out gradient between each iteration
                }
            }
        }
    }
}
