using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Providers.DotNetOp;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// Activation layer. Adds activation functions to a neural net.
    /// </summary>
    public sealed class ActivationLayer : ILayerNew
    {
        readonly Activations.Activation m_activation;

        /// <summary>
        /// Activation layer. Adds activation functions to a neural net.
        /// </summary>
        /// <param name="activation"></param>
        public ActivationLayer(Activations.Activation activation)
        {
            m_activation = activation;
        }

        /// <summary>
        /// 
        /// </summary>
        public Variable Output { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public Variable Input { get; set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="storage"></param>
        /// <param name="training"></param>
        public void Forward(NeuralNetStorage storage, bool training = true)
        {
            Activation.Forward(m_activation,
                Input, Output, storage);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="storage"></param>
        public void Backward(NeuralNetStorage storage)
        {
            Activation.Backward(m_activation, 
                Input, Output, storage);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="storage"></param>
        /// <param name="random"></param>
        /// <param name="initializtion"></param>
        public void Initialize(Variable inputVariable, NeuralNetStorage storage, Random random, 
            Initialization initializtion = Initialization.GlorotUniform)
        {
            UpdateDimensions(inputVariable);
        }

        /// <summary>
        /// Updates the input and output dimensions of the layer
        /// </summary>
        /// <param name="input"></param>
        public void UpdateDimensions(Variable input)
        {
            Input = input;
            Output = Variable.Create(input.Dimensions.ToArray());
        }

        /// <summary>
        /// Copies a minimal version of the layer to be used in a model for predictions.
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="storage"></param>
        /// <param name="copyStorage"></param>
        /// <param name="layers"></param>
        public void Copy(Variable inputVariable, NeuralNetStorage storage, NeuralNetStorage copyStorage, List<ILayerNew> layers)
        {
            var copy = new ActivationLayer(m_activation);
            copy.UpdateDimensions(inputVariable);

            layers.Add(copy);
        }
    }
}
