using System;
using System.Threading.Tasks;
using SharpLearning.Containers.Extensions;
using SharpLearning.Neural.LayersNew;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public static class MaxPool2D
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <param name="desc"></param>
        /// <param name="m_switchX"></param>
        /// <param name="m_switchY"></param>
        /// <param name="training"></param>
        /// <param name="executor"></param>
        public static void Forward(Variable input,
            Variable output, MaxPool2DDescriptor desc,
            int[][] m_switchX, int[][] m_switchY, bool training,
            NeuralNetStorage executor)
        {
            var src = executor.GetTensor(input);
            var dst = executor.GetTensor(output);

            int MB = src.Dimensions[0];
            var IC = src.Dimensions[1];
            var IH = src.Dimensions[2];
            var IW = src.Dimensions[3];

            var OH = dst.Dimensions[2];
            var OW = dst.Dimensions[3];

            var srcData = src.Data;
            var dstData = dst.Data;


            Parallel.For(0, MB, mb =>
            {
                var dstBOffSet = dst.DimensionOffSets[0] * mb;
                var srcBOffSet = src.DimensionOffSets[0] * mb;

                for (int ic = 0; ic < IC; ++ic)
                {
                    var n = ic * OW * OH; // a counter for switches

                    var srcCOffset = srcBOffSet + src.DimensionOffSets[1] * ic;
                    var dstCOffSet = dstBOffSet + dst.DimensionOffSets[1] * ic;

                    for (int oh = 0; oh < OH; ++oh)
                    {
                        var dstHOffSet = dstCOffSet + dst.DimensionOffSets[2] * oh;

                        int hstart = oh * desc.StrideH - desc.PadH;
                        int hend = Math.Min(hstart + desc.PoolH, IH);
                        hstart = Math.Max(hstart, 0);

                        for (int ow = 0; ow < OW; ++ow)
                        {

                            int wstart = ow * desc.StrideW - desc.PadW;
                            int wend = Math.Min(wstart + desc.PoolW, IW);
                            wstart = Math.Max(wstart, 0);

                            var currentMax = double.MinValue;
                            int winx = -1, winy = -1;

                            for (int kh = hstart; kh < hend; ++kh)
                            {
                                var srcHOffSet = srcCOffset + src.DimensionOffSets[2] * kh;

                                for (int kw = wstart; kw < wend; ++kw)
                                {
                                    var srcIndex = srcHOffSet + kw;
                                    var v = srcData[srcIndex];

                                    // perform max pooling and store the index the max location.
                                    if (v > currentMax)
                                    {
                                        currentMax = v;
                                        winx = kw;
                                        winy = kh;
                                    }
                                }
                            }

                            if (training)
                            {
                                m_switchX[mb][n] = winx;
                                m_switchY[mb][n] = winy;
                                n++;
                            }

                            var dstIndex = dstHOffSet + ow;
                            dstData[dstIndex] = currentMax;
                        }
                    }
                }
            });
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <param name="m_switchX"></param>
        /// <param name="m_switchY"></param>
        /// <param name="executor"></param>
        public static void Backward(Variable input, Variable output,
            int[][] m_switchX, int[][] m_switchY,
            NeuralNetStorage executor)
        {
            var inputGradient = executor.GetGradient(input);
            var outputGradient = executor.GetGradient(output);

            int MB = inputGradient.Dimensions[0];
            var IH = inputGradient.Dimensions[2];
            var IW = inputGradient.Dimensions[3];

            var OC = outputGradient.Dimensions[1];
            var OH = outputGradient.Dimensions[2];
            var OW = outputGradient.Dimensions[3];

            inputGradient.Data.Clear();
            var inputGradientData = inputGradient.Data;
            var outputGradientData = outputGradient.Data;

            Parallel.For(0, MB, mb =>
            {
                var switchx = m_switchX[mb];
                var switchy = m_switchY[mb];

                var dstBOffSet = outputGradient.DimensionOffSets[0] * mb;
                var srcBOffSet = inputGradient.DimensionOffSets[0] * mb;

                for (var c = 0; c < OC; c++)
                {
                    var n = c * OW * OH; // conter for switches

                    var srcCOffset = srcBOffSet + inputGradient.DimensionOffSets[1] * c;
                    var dstCOffSet = dstBOffSet + outputGradient.DimensionOffSets[1] * c;

                    for (var h = 0; h < OH; h++)
                    {
                        var dstHOffSet = dstCOffSet + outputGradient.DimensionOffSets[2] * h;

                        for (var w = 0; w < OW; w++)
                        {
                            var dstIndex = dstHOffSet + w;
                            var gradient = outputGradientData[dstIndex];

                            var srcHOffset = srcCOffset + inputGradient.DimensionOffSets[2] * switchy[n];
                            var srcIndex = srcHOffset + switchx[n];
                            inputGradientData[srcIndex] += gradient;

                            n++;
                        }
                    }
                }
            });
        }
    }
}
