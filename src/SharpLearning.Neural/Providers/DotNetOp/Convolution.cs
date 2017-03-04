using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public static class Convolution
    {
        /// <summary>
        /// 
        /// </summary>
        public static void Forward(Tensor<float> input,
            Tensor<float> weights2, Tensor<float> bias2,
            int filterCount, int filterHeight, int filterWidth,
            int strideH, int strideW,
            int padH, int padW,
            Tensor<float> output)
        {
            var src = input.AsTensor4D();
            var weights = weights2.AsTensor4D();
            var bias = bias2.AsTensor1D();
            var dst = output.AsTensor4D();

            //const bool with_groups = false;//conf_.with_groups();

            int G = 1;
            int MB = src.N;
            int OH = dst.H;
            int OW = dst.W;
            int IH = src.H;
            int IW = src.W;

            int OC = dst.C / G;
            int IC = src.C / G;
            int KH = filterHeight; //return cdesc_().weights_desc.dims[2 + with_groups()];
            int KW = filterWidth; //return cdesc_().weights_desc.dims[3 + with_groups()]

            int KSH = strideH;
            int KSW = strideW;

            int padT = padH; //top
            int padL = padW; //left

            for (int mb = 0; mb < MB; ++mb)
            {
                for (int oc = 0; oc < OC; ++oc)
                {
                    for (int oh = 0; oh < OH; ++oh)
                    {
                        for (int ow = 0; ow < OW; ++ow)
                        {
                            var d = Kernel(mb, oc, oh, ow,
                                src, dst, weights,
                                filterHeight, filterWidth,
                                strideH, strideW,
                                padH, padW);

                            d += bias.At(oc);
                            dst.At(mb, oc, oh, ow, d);

                            //data_t & d = dst[dst_d.off(mb, g * OC + oc, oh, ow)];
                            //d = bias ? bias[bias_d.off(g * OC + oc)] : data_t(0);
                            //ker(d, g, mb, oc, oh, ow);
                            //if (with_relu && d < 0) d *= nslope;
                        }
                    }
                }
            }
        }

        static float Kernel(int mb, int oc, int oh, int ow, 
            ITensorIndexer4D<float> src, ITensorIndexer4D<float> dst, ITensorIndexer4D<float> weights,
            int filterHeight, int filterWidth, int strideH, int strideW, int padH, int padW)
        {

            int MB = src.N;
            int OH = dst.H;
            int OW = dst.W;
            int IH = src.H;
            int IW = src.W;

            int OC = dst.C;
            int IC = src.C;
            int KH = filterHeight; //return cdesc_().weights_desc.dims[2 + with_groups()];
            int KW = filterWidth; //return cdesc_().weights_desc.dims[3 + with_groups()]

            int KSH = strideH;
            int KSW = strideW;

            int padT = padH; //top
            int padL = padW; //left

            var result = 0.0f;
            for (int ic = 0; ic < IC; ++ic)
            {
                for (int kh = 0; kh < KH; ++kh)
                {
                    for (int kw = 0; kw < KW; ++kw)
                    {
                        int ih = oh * KSH - padT + kh;
                        int iw = ow * KSW - padL + kw;

                        if (ih < 0 || ih >= IH) continue;
                        if (iw < 0 || iw >= IW) continue;

                        result += src.At(mb, ic, ih, iw)
                         * weights.At(oc, ic, kh, kw);
                    }
                }
            }

            return result;
        }
    }
}
