using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;

namespace SharpLearning.Neural.Layers
{
    /// <summary>
    /// Activation layer. Adds activation functions to a neural net.
    /// </summary>
    [Serializable]
    public class ActivationLayer : ILayer
    {
        /// <summary>
        /// The weights outputted by the layer.
        /// </summary>
        public Matrix<float> OutputActivations;
        Matrix<float> m_inputActivations;

        /// <summary>
        /// Holds the derivative for backward propagation.
        /// </summary>
        public Matrix<float> ActivationDerivative;
        Matrix<float> m_delta;

        IActivation m_activation;

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
        /// Activation layer. Adds activation functions to a neural net.
        /// </summary>
        /// <param name="activation"></param>
        public ActivationLayer(Activation activation)
        {
            ActivationFunc = activation;

            switch (activation)
            {
                case Activation.Undefined:
                    throw new ArgumentException("ActivationLayer must have a defined activation function. Provided with: " + activation);
                case Activation.Relu:
                    m_activation = new ReluActivation();
                    break;
                case Activation.Sigmoid:
                    m_activation = new SigmoidActivation();
                    break;
                default:
                    throw new ArgumentException("Unsupported activation type: " + activation);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="delta"></param>
        /// <returns></returns>
        public Matrix<float> Backward(Matrix<float> delta)
        {
            // Calculate gradient
            ActivationDerivative.Clear();
            m_activation.Derivative(OutputActivations.Data(), ActivationDerivative.Data());

            // Calculate delta for next layer
            delta.PointwiseMultiply(ActivationDerivative, m_delta);

            return m_delta;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix<float> Forward(Matrix<float> input)
        {
            m_inputActivations = input;
            m_inputActivations.CopyTo(OutputActivations); // possible to avoid copy and use input directly?

            m_activation.Activation(OutputActivations.Data());

            return OutputActivations;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="inputDepth"></param>
        /// <param name="batchSize"></param>
        /// <param name="initializtion"></param>
        /// <param name="random"></param>
        public void Initialize(int inputWidth, int inputHeight, int inputDepth, int batchSize, 
            Initialization initializtion, Random random)
        {
            Width = inputWidth;
            Height = inputHeight;
            Depth = inputDepth;

            var fanOut = Width * Height * Depth;

            OutputActivations = Matrix<float>.Build.Dense(batchSize, fanOut);
            ActivationDerivative = Matrix<float>.Build.Dense(batchSize, fanOut);
            m_delta = Matrix<float>.Build.Dense(batchSize, fanOut);
        }

        /// <summary>
        /// The activation layer does not have any parameters or gradients. So adds nothing.
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void AddParameresAndGradients(List<ParametersAndGradients> parametersAndGradients)
        {
            // activation layer does not have any parameters or gradients.
        }

        /// <summary>
        /// Copies a minimal version of the layer to be used in a model for predictions.
        /// </summary>
        /// <param name="layers"></param>
        public void CopyLayerForPredictionModel(List<ILayer> layers)
        {
            var batchSize = 1; // prediction time only uses 1 item at a time.
            var copy = new ActivationLayer(ActivationFunc);

            copy.Width = this.Width;
            copy.Height = this.Height;
            copy.Depth = this.Depth;

            var fanOut = Width * Height * Depth;
            
            copy.OutputActivations = Matrix<float>.Build.Dense(batchSize, fanOut);
            copy.ActivationDerivative = Matrix<float>.Build.Dense(batchSize, fanOut);

            layers.Add(copy);
        }
    }
}
