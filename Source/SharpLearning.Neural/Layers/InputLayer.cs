using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using System;
using System.Collections.Generic;

namespace SharpLearning.Neural.Layers
{
    /// <summary>
    /// 
    /// </summary>
    [Serializable]
    public sealed class InputLayer : ILayer
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
        /// 
        /// </summary>
        /// <param name="inputUnits"></param>
        public InputLayer(int inputUnits)
            :this(1, 1, inputUnits)
        {
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="depth"></param>
        public InputLayer(int width, int height, int depth)
        {
            Width = width;
            Height = height;
            Depth = depth;
            ActivationFunc = Activation.Undefined;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="delta"></param>
        public Matrix<float> Backward(Matrix<float> delta)
        {
            return delta;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        public Matrix<float> Forward(Matrix<float> input)
        {
            return input;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="inputDepth"></param>
        /// <param name="batchSize"></param>
        /// <param name="random"></param>
        public void Initialize(int inputWidth, int inputHeight, int inputDepth, int batchSize, Random random)
        {
            // input layer does not have anything to initialize.
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void AddParameresAndGradients(List<ParametersAndGradients> parametersAndGradients)
        {
            // input layer does not have any parameters or graidents.
        }

        /// <summary>
        /// Input layer does not have any parameters or graidents.
        /// </summary>
        /// <returns></returns>
        public WeightsAndBiases GetGradients()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Input layer does not have any parameters or graidents.
        /// </summary>
        /// <returns></returns>
        public WeightsAndBiases GetParameters()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Copies a minimal version of the layer to be used in a model for predictions.
        /// </summary>
        /// <param name="layers"></param>
        public void CopyLayerForPredictionModel(List<ILayer> layers)
        {
            var copy = new InputLayer(Width, Height, Depth);
            layers.Add(copy);
        }
    }
}
