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
        /// <param name="storage"></param>
        public static void Forward(Activations.Activation activation,
            Variable input, Variable output, NeuralNetStorage storage)
        {
            switch (activation)
            {
                case Activations.Activation.Undefined:
                    break;
                case Activations.Activation.Relu:
                    Relu.Forward(input, output, storage);
                    break;
                case Activations.Activation.SoftMax:
                    SoftMax.Forward(input, output, storage);
                    break;
                case Activations.Activation.MeanSquareError:
                    MeanSquareError.Forward(input, output, storage);
                    break;
                case Activations.Activation.Svm:
                    Svm.Forward(input, output, storage);
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
        /// <param name="storage"></param>
        public static void Backward(SharpLearning.Neural.Activations.Activation activation,
            Variable input, Variable output, NeuralNetStorage storage)
        {
            // calculate gradient
            switch (activation)
            {
                case Activations.Activation.Undefined:
                    break;
                case Activations.Activation.Relu:
                    Relu.Backward(input, output, storage);
                    break;
                case Activations.Activation.SoftMax:
                    SoftMax.Backward(input, output, storage);
                    break;
                case Activations.Activation.MeanSquareError:
                    MeanSquareError.Backward(input, output, storage);
                    break;
                case Activations.Activation.Svm:
                    Svm.Backward(input, output, storage);
                    break;
                default:
                    throw new ArgumentException($"Unsupported activation type {activation}");
            }
        }
    }
}
