using SharpLearning.Containers.Tensors;
using SharpLearning.Containers.Views;
using System.Threading.Tasks;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public static class Dense
    {

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="weights"></param>
        /// <param name="bias"></param>
        /// <param name="output"></param>
        public static void Forward(Tensor<float> input,
            Tensor<float> weights, Tensor<float> bias, 
            Tensor<float> output)
        {
            var srcDimensions = input.Indexer4D;
            var src = input.Create2DIndexer(srcDimensions.DimNCount, srcDimensions.DimXCount * srcDimensions.DimYCount * srcDimensions.DimZCount);
            var dstDimensions = output.Indexer4D;
            var dst = output.Create2DIndexer(dstDimensions.DimNCount, dstDimensions.DimXCount * dstDimensions.DimYCount * dstDimensions.DimZCount); 

            var w = weights.Indexer2D;
            var b = bias.Indexer1D;

            var srcInterval = Interval1D.Create(0, src.DimYCount);
            var wInterval = Interval1D.Create(0, dst.DimYCount); ;

            var srcValues = new float[srcInterval.Length];
            var wValues = new float[wInterval.Length];

            for(int i = 0; i < src.DimXCount; ++i)
            {
                src.RangeX(i, srcInterval, srcValues);
                for (int k = 0; k < w.DimXCount; k++)
                {
                    w.RangeX(k, wInterval, wValues);
                    for (int j = 0; j < dst.DimYCount; j++)
                    {
                        var value = dst.At(i, j);
                        value += srcValues[k] * wValues[j];
                        dst.At(i, j, value);
                    }
                }
            }
        }
    }
}
