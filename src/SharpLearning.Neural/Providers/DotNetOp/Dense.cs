using System.Threading.Tasks;
using SharpLearning.Containers.Tensors;

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
            var src = input.AsTensor4D();
            var src2D = input.AsTensor2D(src.N, src.C * src.H * src.W);

            var dst = output.AsTensor2D();

            var w = weights.AsTensor2D();
            var b = bias.AsTensor1D();

            int MB = dst.H;
            int OC = dst.W;
            int IC = src.C;

            Parallel.For(0, MB, mb =>
            {
                for (int oc = 0; oc < OC; ++oc)
                {
                    var d = Forward(src2D, dst, w, mb, oc);
                    d += b.At(oc);
                    dst.At(mb, oc, d);

                    //data_t* d = &dst[dst_d.off(mb, oc)];
                    //*d = bias ? bias[bias_d.off(oc)] : data_t(0);
                    //if (src_has_spatial)
                    //{
                    //    ker_has_spatial(d, mb, oc);
                    //}
                    //else
                    //{
                    //    ker_no_spatial(d, mb, oc);
                    //}
                }
            });
        }

        static float Forward(ITensorIndexer2D<float> src, ITensorIndexer2D<float> dst,
            ITensorIndexer2D<float> w, int mb, int oc)
        {
            int IC = src.W;

            var d = 0f;
            for (int ic = 0; ic < IC; ++ic)
            {
                d += src.At(mb, ic) * w.At(oc, ic);
            }

            return d;
        }

        static float ForwardSpatial(ITensorIndexer4D<float> src, ITensorIndexer4D<float> dst, 
            ITensorIndexer4D<float> w, int mb, int oc)
        {
            int MB = dst.N;
            int OC = dst.C;
            int IC = src.C;
            int KH = w.H;
            int KW = w.W;

            var d = 0f;
            for (int ic = 0; ic < IC; ++ic)
            {
                for (int kh = 0; kh < KH; ++kh)
                {
                    for (int kw = 0; kw < KW; ++kw)
                    {
                        d += src.At(mb, ic, kh, kw) 
                            * w.At(oc, ic, kh, kw);
                    }
                }
            }

            return d;
        }
    }
}
