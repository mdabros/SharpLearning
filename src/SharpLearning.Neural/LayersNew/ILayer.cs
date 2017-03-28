using System;
using System.Collections.Generic;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.Initializations;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public interface ILayerNew
    {
        /// <summary>
        /// Layer output shape
        /// </summary>
        Variable Output { get; set; }

        /// <summary>
        /// Layer Input shape
        /// </summary>
        Variable Input { get; set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        /// <param name="training"></param>
        void Forward(NeuralNetStorage executor, bool training=true);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        void Backward(NeuralNetStorage executor);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="storage"></param>
        /// <param name="random"></param>
        /// <param name="initializtion"></param>
        void Initialize(Variable inputVariable, NeuralNetStorage storage, Random random, Initialization initializtion);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        void UpdateDimensions(Variable input);
    }
}
