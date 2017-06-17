using System;
using SharpLearning.Neural.LayersNew;

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
            var src = storage.GetTensor(input).Data;
            var dst = storage.GetTensor(output).Data;

            for (int j = 0; j < src.Length; j++)
            {
                dst[j] = ReluMax(src[j]);
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

        static double ReluMax(double input)
        {
            return Math.Max(0, input);
        }

        static double Derivative(double dst, double dstGradient)
        {
            if (dst > 0.0)
                return dstGradient;
            else
                return 0.0f;
        }
    }
}
