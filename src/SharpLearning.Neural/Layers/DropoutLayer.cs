using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Containers.Extensions;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;

namespace SharpLearning.Neural.Layers
{
    /// <summary>
    /// Dropout layer. Dropuout can help reduce overfitting in a neural net.
    /// https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
    /// </summary>
    [Serializable]
    public sealed class DropoutLayer : ILayer
    {
        /// <summary>
        /// Dropout percentage.
        /// </summary>
        public readonly double DropOut;

        Matrix<float> m_dropoutMask;
        Random m_random;

        Matrix<float> m_activations;
        Matrix<float> m_delta;

        /// <summary>
        /// 
        /// </summary>
        public int Width { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public int Height { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public int Depth { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public Activation ActivationFunc { get; set; }

        /// <summary>
        /// Dropout layer for neural network learners.
        /// </summary>
        /// <param name="dropOut">Dropout percentage. The percentage of units randomly omitted during training.
        /// This is a reguralizatin methods for reducing overfitting. Recommended value is 0.5 and range should be between 0.2 and 0.8.
        /// Default (0.0). https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf</param>
        public DropoutLayer(double dropOut = 0.0)
        {
            if (dropOut < 0.0 || dropOut >= 1.0) { throw new ArgumentException("Dropout must be below 1.0 and at least 0.0"); }
            DropOut = dropOut;
            ActivationFunc = Activation.Undefined;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="delta"></param>
        public Matrix<float> Backward(Matrix<float> delta)
        {
            delta.PointwiseMultiply(m_dropoutMask, m_delta);
            m_activations.PointwiseMultiply(m_dropoutMask, m_activations);

            return m_delta;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        public Matrix<float> Forward(Matrix<float> input)
        {
            UpdateDropoutMask();
            input.PointwiseMultiply(m_dropoutMask, m_activations);

            return m_activations;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="inputDepth"></param>
        /// <param name="batchSize"></param>
        /// <param name="initializtion">Initialization type for layers with weights</param>
        /// <param name="random"></param>
        public void Initialize(int inputWidth, int inputHeight, int inputDepth, int batchSize, 
            Initialization initializtion, Random random)
        {
            m_random = new Random(random.Next());
            var fanIn = inputWidth * inputHeight * inputDepth;
            Width = inputWidth;
            Height = inputHeight;
            Depth = inputDepth;

            m_dropoutMask = Matrix<float>.Build.Dense(batchSize, fanIn);
            m_dropoutMask.Data().Map(() => DecideDropOut());

            m_activations = Matrix<float>.Build.Dense(batchSize, fanIn);
            m_delta = Matrix<float>.Build.Dense(batchSize, fanIn);
        }

        float DecideDropOut()
        {
            var dropOutScale = 1.0 / (1.0 - DropOut);
            return (float)(m_random.NextDouble() > DropOut ? dropOutScale : 0.0);
        }

        void UpdateDropoutMask()
        {
            m_dropoutMask.Data().Shuffle(m_random);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void AddParameresAndGradients(List<ParametersAndGradients> parametersAndGradients)
        {
            // dropout layer does not have any parameters or gradients.
        }

        /// <summary>
        /// Copies a minimal version of the layer to be used in a model for predictions.
        /// </summary>
        /// <param name="layers"></param>
        public void CopyLayerForPredictionModel(List<ILayer> layers)
        {
            // dropout layer is not used prediction time.
        }
    }
}
