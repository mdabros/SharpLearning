using System;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public static class Dropout
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="dropoutMask"></param>
        /// <param name="random"></param>
        /// <param name="output"></param>
        /// <param name="executor"></param>
        public static void Forward(Variable input,
            Variable dropoutMask, Random random,
            Variable output, NeuralNetStorage executor)
        {
            var src = executor.GetTensor(input);
            var mask = executor.GetTensor(dropoutMask);
            mask.Data.Shuffle(random);

            var dst = executor.GetTensor(output);
            src.PointwiseMultiply(mask, dst);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="dropoutMask"></param>
        /// <param name="output"></param>
        /// <param name="executor"></param>
        public static void Backward(Variable input,
            Variable dropoutMask,
            Variable output, NeuralNetStorage executor)
        {
            var src = executor.GetTensor(input);
            var mask = executor.GetTensor(dropoutMask);
            var srcDiff = executor.GetGradient(input);
            var dst = executor.GetTensor(output);
            var dstDiff = executor.GetGradient(output);

            dstDiff.PointwiseMultiply(mask, srcDiff);
        }
    }
}
