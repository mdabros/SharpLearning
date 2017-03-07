using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public static class MaxPool
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="poolHeight"></param>
        /// <param name="poolWidth"></param>
        /// <param name="strideH"></param>
        /// <param name="strideW"></param>
        /// <param name="padH"></param>
        /// <param name="padW"></param>
        /// <param name="switches"></param>
        /// <param name="output"></param>
        public static void Forward(Tensor<float> input,
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
    }
}
