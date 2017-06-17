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
        /// <param name="storage"></param>
        public static void Forward(Variable input,
            Variable dropoutMask, Random random,
            Variable output, NeuralNetStorage storage)
        {
            var src = storage.GetTensor(input);
            var mask = storage.GetTensor(dropoutMask);
            mask.Data.Shuffle(random);

            var dst = storage.GetTensor(output);
            src.PointwiseMultiply(mask, dst);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="dropoutMask"></param>
        /// <param name="output"></param>
        /// <param name="storage"></param>
        public static void Backward(Variable input,
            Variable dropoutMask,
            Variable output, NeuralNetStorage storage)
        {
            var src = storage.GetTensor(input);
            var mask = storage.GetTensor(dropoutMask);
            var srcDiff = storage.GetGradient(input);
            var dst = storage.GetTensor(output);
            var dstDiff = storage.GetGradient(output);

            dstDiff.PointwiseMultiply(mask, srcDiff);
        }
    }
}
