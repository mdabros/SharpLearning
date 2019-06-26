using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;

namespace SharpLearning.Neural.Layers
{
    /// <summary>
    /// Fully connected neural network layer.
    /// </summary>
    [Serializable]
    public sealed class DenseLayer : ILayer, IBatchNormalizable
    {
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
        /// Does the layer use batch normalization
        /// </summary>
        public bool BatchNormalization { get; set; }

        /// <summary>
        /// Weights in the layer.
        /// </summary>
        public Matrix<float> Weights;

        /// <summary>
        /// Biases in the layer.
        /// </summary>
        public Vector<float> Bias;

        /// <summary>
        /// Weight gradients.
        /// </summary>
        public Matrix<float> WeightsGradients;

        /// <summary>
        /// Bias gradients.
        /// </summary>
        public Vector<float> BiasGradients;
        
        /// <summary>
        /// Output activation
        /// </summary>
        public Matrix<float> OutputActivations;

        Matrix<float> m_inputActivations;
        Matrix<float> m_delta;

        /// <summary>
        /// Hidden layer for neural network learners.
        /// </summary>
        /// <param name="units">Number of hidden units or neurons in the layer</param>
        /// <param name="activation">Activation function for the layer</param>
        public DenseLayer(int units, Activation activation = Activation.Relu)
        {
            if (units < 1) { throw new ArgumentException("HiddenLayer must have at least 1 hidden unit"); }
            Width = 1;
            Height = 1;
            Depth = units;
            ActivationFunc = activation;
        }

        /// <summary>
        /// Backward pass.
        /// </summary>
        /// <param name="delta"></param>
        public Matrix<float> Backward(Matrix<float> delta)
        {
            // calculate gradients
            m_inputActivations.TransposeThisAndMultiply(delta, WeightsGradients);
            delta.SumColumns(BiasGradients);
            
            // calculate delta for next layer
            delta.TransposeAndMultiply(Weights, m_delta);
            
            return m_delta;
        }

        /// <summary>
        /// Forward pass.
        /// </summary>
        /// <param name="input"></param>
        public Matrix<float> Forward(Matrix<float> input)
        {
            m_inputActivations = input;

            input.Multiply(Weights, OutputActivations);
            OutputActivations.AddRowWise(Bias, OutputActivations);

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
            var fans = WeightInitialization.GetFans(this, inputWidth, inputHeight, inputDepth);
            var distribution = WeightInitialization.GetWeightDistribution(initializtion, fans, random);
            
            Weights = Matrix<float>.Build.Random(fans.FanIn, fans.FanOut, distribution);
            Bias = Vector<float>.Build.Dense(fans.FanOut, 0.0f);

            WeightsGradients = Matrix<float>.Build.Dense(fans.FanIn, fans.FanOut);
            BiasGradients = Vector<float>.Build.Dense(fans.FanOut);

            OutputActivations = Matrix<float>.Build.Dense(batchSize, fans.FanOut);

            m_delta = Matrix<float>.Build.Dense(batchSize, fans.FanIn);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void AddParameresAndGradients(List<ParametersAndGradients> parametersAndGradients)
        {
            var weights = new ParametersAndGradients(Weights.Data(), WeightsGradients.Data());
            var bias = new ParametersAndGradients(Bias.Data(), BiasGradients.Data());
            parametersAndGradients.Add(weights);
            parametersAndGradients.Add(bias);
        }

        /// <summary>
        /// Copies a minimal version of the layer to be used in a model for predictions.
        /// </summary>
        /// <param name="layers"></param>
        public void CopyLayerForPredictionModel(List<ILayer> layers)
        {
            var batchSize = 1; // prediction time only uses 1 item at a time.
            var fanOut = Width * Height * Depth;

            var copy = new DenseLayer(fanOut, ActivationFunc);

            copy.Weights = Matrix<float>.Build.Dense(Weights.RowCount, Weights.ColumnCount);
            copy.Bias = Vector<float>.Build.Dense(Bias.Count);

            Array.Copy(Weights.Data(), copy.Weights.Data(), Weights.Data().Length);
            Array.Copy(Bias.Data(), copy.Bias.Data(), Bias.Data().Length);

            copy.OutputActivations = Matrix<float>.Build.Dense(batchSize, fanOut);

            layers.Add(copy);
        }
    }
}
