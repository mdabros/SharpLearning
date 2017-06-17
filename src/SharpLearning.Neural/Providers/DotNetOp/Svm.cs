using System;
using SharpLearning.Neural.LayersNew;
using SharpLearning.Containers.Extensions;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// square error operator.
    /// </summary>
    public static class Svm
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
            var dst = storage.GetTensor(output); // svm outputs, predictions
            var dstDiff = storage.GetGradient(output); // target values are storred in output gradient.
            var srcDiff = storage.GetGradient(input);

            if (dst.Shape != dstDiff.Shape)
            {
                throw new ArgumentException($"output shape: {dst.Shape} differs from input shape {dstDiff.Shape}");
            }

            // Assumes 2D and collapse to 2D.
            const double margin = 1.0;
            var rows = dstDiff.Dimensions[0];
            var cols = dstDiff.DimensionOffSets[0];

            var targetsData = dstDiff.Data;
            var predictionsData = dst.Data;
            var srcDiffData = srcDiff.Data;
            srcDiffData.Clear(); // clear results in case of re-use,

            for (int batchItem = 0; batchItem < rows; batchItem++)
            {
                var maxTarget = 0.0;
                var maxTargetIndex = 0;
                var rowOffSet = batchItem * cols;
                for (int col = 0; col < cols; col++)
                {
                    var index = rowOffSet + col;
                    var targetValue = targetsData[index];
                    if (targetValue > maxTarget)
                    {
                        maxTarget = targetValue;
                        maxTargetIndex = col;
                    }
                }

                var maxPredictionIndex = rowOffSet + maxTargetIndex;
                var maxTargetScore = predictionsData[maxPredictionIndex];
                for (int i = 0; i < cols; i++)
                {
                    if (i == maxTargetIndex) { continue; }

                    // The score of the target should be higher than he score of any other class, by a margin
                    var index = rowOffSet + i;
                    var diff = -maxTargetScore + predictionsData[index] + margin;
                    if (diff > 0)
                    {
                        srcDiffData[index] += 1;
                        srcDiffData[maxPredictionIndex] -= 1;
                    }
                }
            }
        }
    }
}
