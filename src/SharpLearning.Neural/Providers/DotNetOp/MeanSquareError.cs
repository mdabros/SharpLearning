using System;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// square error operator.
    /// </summary>
    public static class MeanSquareError
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <param name="storage"></param>
        public static void Forward(Variable input, Variable output, NeuralNetStorage storage)
        {
            var src = storage.GetTensor(input);
            var dst = storage.GetTensor(output);
                        
            var srcData = src.Data;
            var dstData = dst.Data;

            if(src.Shape != dst.Shape)
            {
                throw new ArgumentException($"input shape: {src.Shape} differs from output shape {dst.Shape}");
            }

            // do nothing, output raw scores
            srcData.CopyTo(dstData, 0);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output">In softmax output gradient will be the original target values</param>
        /// <param name="storage"></param>
        public static void Backward(Variable input, Variable output, NeuralNetStorage storage)
        {
            var dst = storage.GetTensor(output); // mse outputs, predictions
            var dstDiff = storage.GetGradient(output); // target values are storred in output gradient.
            var srcDiff = storage.GetGradient(input);

            if (dst.Shape != dstDiff.Shape)
            {
                throw new ArgumentException($"output shape: {dst.Shape} differs from target shape {dstDiff.Shape}");
            }

            dst.Subtract(dstDiff, srcDiff);
        }
    }
}
