using System;
using System.Linq;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Providers.DotNetOp;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class ActivationLayer : ILayerNew
    {
        readonly Activations.Activation m_activation;

        /// <summary>
        /// 
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
        /// <param name="executor"></param>
        public void Forward(Executor executor)
        {
            Activation.Forward(m_activation,
                Input, Output, executor);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        public void Backward(Executor executor)
        {
            Activation.Backward(m_activation, 
                Input, Output, executor);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="excecutor"></param>
        /// <param name="random"></param>
        /// <param name="initializtion"></param>
        public void Initialize(Variable inputVariable, Executor excecutor, Random random, 
            Initialization initializtion = Initialization.GlorotUniform)
        {
            Input = inputVariable;
            Output = Variable.Create(inputVariable.Dimensions.ToArray());
        }
    }
}
