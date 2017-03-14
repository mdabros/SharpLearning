using System;
using System.Numerics;
using System.Threading.Tasks;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public class Convolution
    {
        readonly int m_filterCount;
        readonly int m_filterH;
        readonly int m_filterW;
        readonly int m_strideH;
        readonly int m_strideW;
        readonly int m_padH;
        readonly int m_padW;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="filterCount">Number of filters</param>
        /// <param name="filterH">The height of each filter</param>
        /// <param name="filterW">The width of each filter</param>
        /// <param name="strideH">The vertical stride of the filter</param>
        /// <param name="strideW">The horizontal stride of the filter</param>
        /// <param name="padH">Zero padding at the top and bottom</param>
        /// <param name="padW">Zero padding to the left and right</param>
        public Convolution(int filterCount, int filterH, int filterW, 
            int strideH, int strideW, 
            int padH, int padW)
        {
            if (filterCount < 1)
            { throw new ArgumentException($"filterCount must be at least 1, was {filterCount}"); }
            if (filterH < 1)
            { throw new ArgumentException($"filterH must be at least 1, was {filterH}"); }
            if (filterW < 1)
            { throw new ArgumentException($"filterW must be at least 1, was {filterW}"); }
            if (strideH < 1)
            { throw new ArgumentException($"strideH must be at least 1, was {strideH}"); }
            if (strideW < 1)
            { throw new ArgumentException($"strideW must be at least 1, was {strideW}"); }
            if (padH < 0)
            { throw new ArgumentException($"padH must be at least 0, was {padH}"); }
            if (padW < 0)
            { throw new ArgumentException($"padW must be at least 0, was {padW}"); }

            m_filterCount = filterCount;
            m_filterH = filterH;
            m_filterW = filterW;
            m_strideH = strideH;
            m_strideW = strideW;
            m_padH = padH;
            m_padW = padW;
        }

        /// <summary>
        /// 
        /// </summary>
        public void Forward(Tensor<float> input,
            Tensor<float> weights, Tensor<float> bias,
            Tensor<float> output)
        {
            var src = input;
            var dst = output;

            //const bool with_groups = false;//conf_.with_groups();

            int MB = src.Dimensions[0];
            int OH = dst.Dimensions[2];
            int OW = dst.Dimensions[3];
            int IH = src.Dimensions[2];
            int IW = src.Dimensions[3];

            int OC = dst.Dimensions[1];
            int IC = src.Dimensions[1];
            int KH = m_filterH; //return cdesc_().weights_desc.dims[2 + with_groups()];
            int KW = m_filterW; //return cdesc_().weights_desc.dims[3 + with_groups()]

            int KSH = m_strideH;
            int KSW = m_strideW;

            int padT = m_padH; //top
            int padL = m_padW; //left

            var dstData = dst.Data;
            var biasData = bias.Data;

            Parallel.For(0, MB, mb =>
            //for (int mb = 0; mb < MB; mb++)
            {
                var dstBOffSet = dst.DimensionOffSets[0] * mb;
                for (int oc = 0; oc < OC; ++oc)
                {
                    var dstCOffSet = dstBOffSet + dst.DimensionOffSets[1] * oc;
                    for (int oh = 0; oh < OH; ++oh)
                    {
                        var dstHffSet = dstCOffSet + dst.DimensionOffSets[2] * oh;
                        for (int ow = 0; ow < OW; ++ow)
                        {
                            var d = Kernel(mb, oc, oh, ow,
                                src, weights,
                                m_filterH, m_filterW,
                                m_strideH, m_strideW,
                                m_padH, m_padW);

                            d += biasData[oc];

                            dstData[dstHffSet + ow] = d;
                            //if (with_relu && d < 0) d *= nslope;
                        }
                    }
                }
            });
        }

        static float Kernel(int mb, int oc, int oh, int ow, 
            Tensor<float> src, Tensor<float> weights,
            int filterHeight, int filterWidth, int strideH, int strideW, int padH, int padW)
        {

            int IH = src.Dimensions[2];
            int IW = src.Dimensions[3];

            int IC = src.Dimensions[1];
            int KH = filterHeight; //return cdesc_().weights_desc.dims[2 + with_groups()];
            int KW = filterWidth; //return cdesc_().weights_desc.dims[3 + with_groups()]

            int KSH = strideH;
            int KSW = strideW;

            int padT = padH; //top
            int padL = padW; //left

            var result = 0.0f;

            var srcData = src.Data;
            var srcBOffSet = mb * src.DimensionOffSets[0];

            var wData = weights.Data;
            var wBOffSet = oc * weights.DimensionOffSets[0];

            for (int ic = 0; ic < IC; ++ic)
            {
                var srcCOffSet = srcBOffSet + src.DimensionOffSets[1] * ic;
                var wCOffSet = wBOffSet + weights.DimensionOffSets[1] * ic;

                for (int kh = 0; kh < KH; ++kh)
                {
                    int ih = oh * KSH - padT + kh;
                    if (ih < 0 || ih >= IH) continue;

                    var srcHOffSet = srcCOffSet + src.DimensionOffSets[2] * ih;
                    var wHOffSet = wCOffSet + weights.DimensionOffSets[2] * kh;

                    if(Vector.IsHardwareAccelerated)
                    {
                        result = InnerLoopSimd(ow, IW, KW, KSW, padL,
                            srcData, wData, srcHOffSet, wHOffSet);

                    }
                    else
                    {
                        result = InnerLoop(ow, IW, KW, KSW, padL,
                            srcData, wData, srcHOffSet, wHOffSet);

                    }
                }
            }

            return result;
        }

        static float InnerLoop(int ow, int IW, int KW, int KSW, int padL, 
            float[] srcData, float[] wData, int srcHOffSet, int wHOffSet)
        {
            var result = 0f;
            for (int kw = 0; kw < KW; ++kw)
            {
                int iw = ow * KSW - padL + kw;
                if (iw < 0 || iw >= IW) continue;

                result += srcData[srcHOffSet + iw] * wData[wHOffSet + kw];
                //result += src.At(mb, ic, ih, iw)
                //    * weights.At(oc, ic, kh, kw);
            }

            return result;
        }

        static float InnerLoopSimd(int ow, int IW, int KW, int KSW, int padL, 
            float[] srcData, float[] wData, int srcHOffSet, int wHOffSet)
        {
            var simdLength = Vector<float>.Count;
            var kw = 0;

            var result = 0f;

            for (kw = 0; kw <= KW - simdLength; kw += simdLength)
            {
                int iw = ow * KSW - padL + kw;
                //if (iw < 0 || iw >= IW) continue; issue

                var vSrc = new Vector<float>(srcData, srcHOffSet + iw);
                var vW = new Vector<float>(wData, wHOffSet + kw);
                result += Vector.Dot(vSrc, vW);
            }

            for (; kw < KW; ++kw)
            {
                int iw = ow * KSW - padL + kw;
                if (iw < 0 || iw >= IW) continue;

                result += srcData[srcHOffSet + iw] * wData[wHOffSet + kw];
            }

            return result;
        }
    }
}
