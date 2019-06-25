using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;

namespace SharpLearning.Neural.Layers
{
    /// <summary>
    /// Interface for neural net layer
    /// </summary>
    public interface ILayer
    {
        /// <summary>
        /// Width of this layer
        /// </summary>
        int Width { get;  }

        /// <summary>
        /// Height of this layer
        /// </summary>
        int Height { get; }

        /// <summary>
        /// Depth og this layer
        /// </summary>
        int Depth { get; }

        /// <summary>
        /// Activation
        /// </summary>
        Activation ActivationFunc { get; set; }

        /// <summary>
        /// Backward pass.
        /// </summary>
        /// <param name="delta"></param>
        Matrix<float> Backward(Matrix<float> delta);

        /// <summary>
        /// Forward pass.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        Matrix<float> Forward(Matrix<float> input);

        /// <summary>
        /// Initialize.
        /// </summary>
        /// <param name="inputWidth">Width of the previous layer</param>
        /// <param name="inputHeight">Height of the previous layer</param>
        /// <param name="inputDepth">Depth of the previous layer</param>
        /// <param name="batchSize">batch size</param>
        /// <param name="initializtion">Initialization type for layers with weights</param>
        /// <param name="random"></param>
        void Initialize(int inputWidth, int inputHeight, int inputDepth, int batchSize, 
            Initialization initializtion, Random random);

        /// <summary>
        /// Adds the layers parameters and gradients (if any) to the list.
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        void AddParameresAndGradients(List<ParametersAndGradients> parametersAndGradients);

        /// <summary>
        /// Copies a minimal version of the layer to be used in a model for predictions.
        /// </summary>
        /// <param name="layers"></param>
        void CopyLayerForPredictionModel(List<ILayer> layers);
    }
}