using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public static class SoftMax
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

            if (src.Shape != dst.Shape)
            {
                throw new ArgumentException($"output shape: {dst.Shape} differs from input shape {src.Shape}");
            }

            // Assumes 2D and collapse to 2D.
            // assumes src and dst has same dimension.
            var srcData = src.Data;
            var dstData = dst.Data;
            var rows = src.Dimensions[0];
            var cols = src.DimensionOffSets[0];

            for (int row = 0; row < rows; row++)
            {
                var rowSum = 0.0;
                var max = double.MinValue;

                for (int col = 0; col < cols; ++col)
                {
                    var index = row * cols + col;
                    var value = srcData[index];
                    if (value > max)
                    {
                        max = value;
                    }
                }

                for (int col = 0; col < cols; ++col)
                {
                    var index = row * cols + col;

                    var value = Math.Exp(srcData[index] - max);
                    rowSum += value;
                    dstData[index] = value;
                }

                for (int col = 0; col < cols; ++col)
                {
                    var index = row * cols + col;
                    dstData[index] = dstData[index] / rowSum;
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output">In softmax output gradient will be the original target values</param>
        /// <param name="storage"></param>
        public static void Backward(Variable input, Variable output, NeuralNetStorage storage)
        {
            var dst = storage.GetTensor(output); // softmax outputs, predictions
            var dstDiff = storage.GetGradient(output); // target values are storred in output gradient.
            var srcDiff = storage.GetGradient(input);

            if (dst.Shape != srcDiff.Shape)
            {
                throw new ArgumentException($"output shape: {dst.Shape} differs from input shape {srcDiff.Shape}");
            }

            dstDiff.Subtract(dst, srcDiff);
            srcDiff.Map(v => v * -1);
        }
    }
}
