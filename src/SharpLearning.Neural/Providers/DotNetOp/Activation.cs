using System;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public static class Activation
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="activation"></param>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <param name="executor"></param>
        public static void Forward(Activations.Activation activation,
            Variable input, Variable output, NeuralNetStorage executor)
        {
            switch (activation)
            {
                case Activations.Activation.Undefined:
                    break;
                case Activations.Activation.Relu:
                    Relu.Forward(input, output, executor);
                    break;
                case Activations.Activation.SoftMax:
                    SoftMax.Forward(input, output, executor);
                    break;
                default:
                    throw new ArgumentException($"Unsupported activation type {activation}");
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="activation"></param>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <param name="executor"></param>
        public static void Backward(SharpLearning.Neural.Activations.Activation activation,
            Variable input, Variable output, NeuralNetStorage executor)
        {
            // calculate gradient
            switch (activation)
            {
                case Activations.Activation.Undefined:
                    break;
                case Activations.Activation.Relu:
                    Relu.Backward(input, output, executor);
                    break;
                case Activations.Activation.SoftMax:
                    SoftMax.Backward(input, output, executor);
                    break;
                default:
                    throw new ArgumentException($"Unsupported activation type {activation}");
            }
        }
    }
}
