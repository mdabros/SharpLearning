using System;
using System.Linq;
using System.Threading.Tasks;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public class MaxPool
    {
        readonly int m_poolH;
        readonly int m_poolW;
        readonly int m_strideH;
        readonly int m_strideW;
        readonly int m_padH;
        readonly int m_padW;

        readonly int[][] m_switchX;
        readonly int[][] m_switchY;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="poolH">Height of the pooling window</param>
        /// <param name="poolW">Width of the pooling window</param>
        /// <param name="strideH">Pooling vertical stride</param>
        /// <param name="strideW">Pooling horizontal stride</param>
        /// <param name="padH">Size of vertical padding</param>
        /// <param name="padW">Size of horizontal padding</param>
        /// <param name="outN">Number of batch items in output</param>
        /// <param name="outC">Depth of output</param>
        /// <param name="outH">Height of output</param>
        /// <param name="outw">Width of output</param>
        public MaxPool(int poolH, int poolW,
            int strideH, int strideW,
            int padH, int padW, 
            int outN, int outC, int outH, int outw)
        {
            if (poolH < 1)
            { throw new ArgumentException($"filterH must be at least 1, was {poolH}"); }
            if (poolW < 1)
            { throw new ArgumentException($"filterW must be at least 1, was {poolW}"); }
            if (strideH < 1)
            { throw new ArgumentException($"strideH must be at least 1, was {strideH}"); }
            if (strideW < 1)
            { throw new ArgumentException($"strideW must be at least 1, was {strideW}"); }
            if (padH < 0)
            { throw new ArgumentException($"padH must be at least 0, was {padH}"); }
            if (padW < 0)
            { throw new ArgumentException($"padW must be at least 0, was {padW}"); }

            m_poolH = poolH;
            m_poolW = poolW;
            m_strideH = strideH;
            m_strideW = strideW;
            m_padH = padH;
            m_padW = padW;

            m_switchX = Enumerable.Range(0, outN).Select(v => new int[outC * outH * outw]).ToArray();
            m_switchY = Enumerable.Range(0, outN).Select(v => new int[outC * outH * outw]).ToArray();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        public void Forward(Tensor<float> input,
            Tensor<float> output)
        {
            var src = input;
            var dst = output;

            int MB = src.Dimensions[0];

            Parallel.For(0, MB, mb =>
            {
                ForwardSingleItem(input, output, mb);
            });
        }

        void ForwardSingleItem(Tensor<float> src, Tensor<float> dst, int mb)
        {
            var MB = src.Dimensions[0];
            var IC = src.Dimensions[1];
            var IH = src.Dimensions[2];
            var IW = src.Dimensions[3];

            var OH = dst.Dimensions[2];
            var OW = dst.Dimensions[3];

            var srcData = src.Data;
            var dstData = dst.Data;

            var dstBOffSet = dst.DimensionOffSets[0] * mb;
            var srcBOffSet = src.DimensionOffSets[0] * mb;

            var switchX = m_switchX[mb];
            var switchY = m_switchY[mb];

            for (int ic = 0; ic < IC; ++ic)
            {
                var n = ic * OW * OH; // a counter for switches

                var srcCOffset = srcBOffSet + src.DimensionOffSets[1] * ic;
                var dstCOffSet = dstBOffSet + dst.DimensionOffSets[1] * ic;

                for (int oh = 0; oh < OH; ++oh)
                {
                    var dstHOffSet = dstCOffSet + dst.DimensionOffSets[2] * oh;

                    int hstart = oh * m_strideH - m_padH;
                    int hend = Math.Min(hstart + m_poolH, IH);
                    hstart = Math.Max(hstart, 0);

                    for (int ow = 0; ow < OW; ++ow)
                    {

                        int wstart = ow * m_strideW - m_padW;
                        int wend = Math.Min(wstart + m_poolW, IW);
                        wstart = Math.Max(wstart, 0);

                        var currentMax = float.MinValue;
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

                        switchX[n] = winx;
                        switchY[n] = winy;
                        n++;

                        var dstIndex = dstHOffSet + ow;
                        dstData[dstIndex] = currentMax;
                    }
                }
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        public void Backward(Tensor<float> input,
            Tensor<float> output)
        {
            var src = input;
            var dst = output;

            int MB = src.Dimensions[0];

            // enumerate each batch item one at a time
            Parallel.For(0, MB, mb =>
            {
                BackwardSingleItem(input, output, mb);
            });
        }

        void BackwardSingleItem(Tensor<float> inputGradient, Tensor<float> outputGradient, int mb)
        {
            var MB = inputGradient.Dimensions[0];
            var IH = inputGradient.Dimensions[2];
            var IW = inputGradient.Dimensions[3];

            var OC = outputGradient.Dimensions[1];
            var OH = outputGradient.Dimensions[2];
            var OW = outputGradient.Dimensions[3];

            var inputData = inputGradient.Data;
            var outputData = outputGradient.Data;

            var switchx = m_switchX[mb];
            var switchy = m_switchY[mb];

            var dstBOffSet = outputGradient.DimensionOffSets[0] * mb;
            var srcBOffSet = inputGradient.DimensionOffSets[0] * mb;

            for (var c = 0; c < OC; c++)
            {
                var n = c * OW * OH; // conter for switches
                var dstCOffSet = dstBOffSet + outputGradient.DimensionOffSets[1] * c;

                var srcCOffset = srcBOffSet + inputGradient.DimensionOffSets[1] * c;

                for (var h = 0; h < OH; h++)
                {
                    var dstHOffSet = dstCOffSet + outputGradient.DimensionOffSets[2] * h;

                    for (var w = 0; w < OW; w++)
                    {
                        var dstIndex = dstHOffSet + w;
                        var gradient = outputData[dstIndex];

                        var srcHOffset = srcCOffset + inputGradient.DimensionOffSets[2] * switchy[n];
                        var srcIndex = srcHOffset + switchx[n];
                        inputData[srcIndex] += gradient;

                        n++;
                    }
                }
            }
        }

            #region MKL IMPL

        static void Forward(Tensor<float> input,
            int poolHeight, int poolWidth,
            int strideH, int strideW,
            int padH, int padW,
            int[] switches,
            Tensor<float> output)
        {
            var src = input;
            var dst = output;

            int MB = src.Dimensions[0];
            int OC = src.Dimensions[1];
            int OH = dst.Dimensions[2];
            int OW = dst.Dimensions[3];

            var dstData = dst.Data;
            //for (int mb = 0; mb < MB; ++mb)
            Parallel.For(0, MB, mb =>
            {
                var dstBOffSet = dst.DimensionOffSets[0] * mb;
                for (int oc = 0; oc < OC; ++oc)
                {
                    var dstcOffSet = dstBOffSet + dst.DimensionOffSets[1] * oc;
                    for (int oh = 0; oh < OH; ++oh)
                    {
                        var dsthOffSet = dstcOffSet + dst.DimensionOffSets[2] * oh;
                        for (int ow = 0; ow < OW; ++ow)
                        {
                            var dstIndex = dsthOffSet + ow;
                            dstData[dstIndex] = MaxPoolForward(input, mb, oc, oh, ow,
                                poolHeight, poolWidth,
                                strideH, strideW,
                                padH, padW, switches);
                        }
                    }
                }
            });
        }

        static float MaxPoolForward(Tensor<float> input, int mb, int oc, int oh, int ow,
            int poolHeight, int poolWidth,
            int strideHeight, int strideWidth,
            int padHeight, int padWidth,
            int[] switches)
        {
            int IH = input.Dimensions[2];
            int IW = input.Dimensions[3];
            int KH = poolHeight;
            int KW = poolWidth;
            int SH = strideHeight;
            int SW = strideWidth;
            int padT = padHeight;
            int padL = padWidth;

            float d = float.NegativeInfinity;
            var src = input;
            var srcData = input.Data;

            var srcOffSet = mb * src.DimensionOffSets[0] + oc * src.DimensionOffSets[1];

            for (int kh = 0; kh < KH; ++kh)
            {
                int ih = oh * SH - padT + kh;
                var srchOffSet = srcOffSet + src.DimensionOffSets[2] * ih;

                for (int kw = 0; kw < KW; ++kw)
                {
                    int iw = ow * SW - padL + kw;

                    if (ih < 0 || ih >= IH) continue;
                    if (iw < 0 || iw >= IW) continue;

                    var srcIndex = srchOffSet + iw;
                    var s = srcData[srcIndex];

                    if (s > d)
                    {
                        d = s;
                        switches[srcIndex] = kh * KW + kw;
                    }
                }
            }

            return d;
        }

        # endregion
    }
}
