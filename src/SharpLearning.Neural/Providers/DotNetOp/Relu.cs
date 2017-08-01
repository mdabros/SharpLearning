using System;
using SharpLearning.Neural.LayersNew;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public static class Relu
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <param name="storage"></param>
        public static void Forward(Variable input, Variable output, NeuralNetStorage storage)
        {
            var inputTensor = storage.GetTensor(input);
            var outputTensor = storage.GetTensor(output);

            Forward(inputTensor, outputTensor);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        public static void Forward(Tensor<double> input, Tensor<double> output)
        {
            var inputData = input.Data;
            var outputData = output.Data;

            for (int j = 0; j < inputData.Length; j++)
            {
                outputData[j] = ReluMax(inputData[j]);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <param name="strorage"></param>
        public static void Backward(Variable input, Variable output, NeuralNetStorage strorage)
        {
            var outputTensor = strorage.GetTensor(output);
            var outputGradient = strorage.GetGradient(output);
            var inputGradient = strorage.GetGradient(input);

            Backward(outputTensor, outputGradient, inputGradient);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="output"></param>
        /// <param name="outputGradient"></param>
        /// <param name="inputGradient"></param>
        public static void Backward(Tensor<double> output, Tensor<double> outputGradient, Tensor<double> inputGradient)
        {
            var outputData = output.Data;
            var outputGradientData = outputGradient.Data;
            var inputGradientData = inputGradient.Data;

            for (int j = 0; j < outputData.Length; j++)
            {
                inputGradientData[j] = Derivative(outputData[j], outputGradientData[j]);
            }
        }

        static double ReluMax(double input)
        {
            return Math.Max(0, input);
        }

        static double Derivative(double output, double outputGradient)
        {
            if (output > 0.0)
                return outputGradient;
            else
                return 0.0f;
        }
    }
}
