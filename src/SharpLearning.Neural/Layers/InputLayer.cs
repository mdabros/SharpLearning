using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;

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
            if (width < 1) { throw new ArgumentException("width is less than 1: " + width); }
            if (height < 1) { throw new ArgumentException("height is less than 1: " + height); }
            if (depth < 1) { throw new ArgumentException("depth is less than 1: " + depth); }

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
        /// <param name="initializtion"></param>
        /// <param name="random"></param>

        public void Initialize(int inputWidth, int inputHeight, int inputDepth, int batchSize, 
            Initialization initializtion, Random random)
        {
            // input layer does not have anything to initialize.
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void AddParameresAndGradients(List<ParametersAndGradients> parametersAndGradients)
        {
            // input layer does not have any parameters or gradients.
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
