using System;

namespace SharpLearning.Neural.Test.RefactorBranStorm
{
    public static class Operators
    {
        public static class ReLU
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

                for (int j = 0; j < inputTensor.Data.Length; j++)
                {
                    outputTensor.Data[j] = ReluMax(inputTensor.Data[j]);
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

                for (int j = 0; j < outputTensor.Data.Length; j++)
                {
                    inputGradient.Data[j] = Derivative(outputTensor.Data[j], outputGradient.Data[j]);
                }
            }

            static float ReluMax(float input)
            {
                return Math.Max(0, input);
            }

            static float Derivative(float output, float outputGradient)
            {
                if (output > 0.0)
                    return outputGradient;
                else
                    return 0.0f;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public static class Sigmoid
        {
            /// <summary>
            /// 
            /// </summary>
            /// <param name="input"></param>
            /// <param name="output"></param>
            /// <param name="storage"></param>
            public static void Forward(Variable input, Variable output, NeuralNetStorage storage)
            {
                var src = storage.GetTensor(input).Data;
                var dst = storage.GetTensor(output).Data;

                for (int j = 0; j < src.Length; j++)
                {
                    dst[j] = DoSigmoid(src[j]);
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
                var dst = strorage.GetTensor(output).Data;
                var dstDiff = strorage.GetGradient(output).Data;
                var srcDiff = strorage.GetGradient(input).Data;

                for (int j = 0; j < dst.Length; j++)
                {
                    srcDiff[j] = Derivative(dst[j], dstDiff[j]);
                }
            }

            static float DoSigmoid(float input)
            {
                return 1.0f / (1.0f + (float)Math.Exp(-input));
            }

            static float Derivative(float dst, float dstGradient)
            {
                return dst * (1.0f - dst) * dstGradient;
            }
        }

        public static class Dense
        {
            /// <summary>
            /// 
            /// </summary>
            /// <param name="input"></param>
            /// <param name="weights"></param>
            /// <param name="bias"></param>
            /// <param name="output"></param>
            /// <param name="storage"></param>
            public static void Forward(Variable input,
                Variable weights, Variable bias,
                Variable output, NeuralNetStorage storage)
            {
                var src = storage.GetTensor(input);

                var w = storage.GetTensor(weights);
                var b = storage.GetTensor(bias).Data;

                var dst = storage.GetTensor(output);

                src.Multiply(w, dst);
                dst.AddRowWise(b, dst);
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="input"></param>
            /// <param name="weights"></param>
            /// <param name="bias"></param>
            /// <param name="output"></param>
            /// <param name="storage"></param>
            public static void Backward(Variable input,
                Variable weights, Variable bias,
                Variable output, NeuralNetStorage storage)
            {
                var src = storage.GetTensor(input);
                var srcDiff = storage.GetGradient(input);

                var w = storage.GetTensor(weights);
                var wDiff = storage.GetGradient(weights);

                var bDiff = storage.GetGradient(bias).Data;
                var dstDiff = storage.GetGradient(output);

                // calculate gradients
                src.TransposeThisAndMultiply(dstDiff, wDiff);
                dstDiff.SumColumns(bDiff);

                // calculate delta for next layer
                dstDiff.TransposeAndMultiply(w, srcDiff);
            }
        }
    }
}
