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
        /// <param name="executor"></param>
        public static void Forward(Variable input, Variable output, NeuralNetStorage executor)
        {
            var src = executor.GetTensor(input);
            var dst = executor.GetTensor(output);

            // Assumes 2D and collapse to 2D.
            // assumes src and dst has same dimension.
            var srcData = src.Data;
            var dstData = dst.Data;
            var rows = src.Dimensions[0];
            var cols = src.DimensionOffSets[0];

            for (int row = 0; row < rows; row++)
            {
                var rowSum = 0.0f;
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

                    var value = (float)Math.Exp(srcData[index] - max);
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
        /// <param name="executor"></param>
        public static void Backward(Variable input, Variable output, NeuralNetStorage executor)
        {
            var dst = executor.GetTensor(output); // softmax outputs, predictions
            var dstDiff = executor.GetGradient(output); // target values are storred in output gradient.
            var srcDiff = executor.GetGradient(input);

            dstDiff.Subtract(dst, srcDiff);
            srcDiff.Map(v => v * -1);
        }
    }
}
