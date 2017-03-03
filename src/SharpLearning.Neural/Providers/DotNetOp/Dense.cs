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
            var src = input.Indexer4D;
            var src2D = input.Create2DIndexer(src.DimNCount, src.DimXCount * src.DimYCount * src.DimZCount);

            var dst = output.Indexer4D;
            

            var w = weights.Indexer2D;
            var b = bias.Indexer1D;

            int MB = src.DimNCount;
            int OC = dst.DimZCount;
            int IC = src.DimZCount;

            var dst2D = output.Create2DIndexer(MB, OC);

            Parallel.For(0, MB, mb =>
            //for (int mb = 0; mb < MB; ++mb)
            {
                for (int oc = 0; oc < OC; ++oc)
                {
                    var d = Forward(src2D, dst, w, mb, oc);
                    d += b.At(oc);
                    dst2D.At(mb, oc, d);

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

        static float Forward(ITensorIndexer2D<float> src, ITensorIndexer4D<float> dst,
            ITensorIndexer2D<float> w, int mb, int oc)
        {
            int MB = src.DimXCount;
            int OC = dst.DimZCount;
            int IC = src.DimYCount;
            int KH = w.DimYCount;
            int KW = w.DimXCount;
            var d = 0f;

            for (int ic = 0; ic < IC; ++ic)
            {
                d += src.At(mb, ic) * w.At(ic, oc);
                //d += src[src_d.off(mb, ic)] * weights[weights_d.off(oc, ic)];
            }

            return d;
        }

        static float ForwardSpatial(ITensorIndexer4D<float> src, ITensorIndexer4D<float> dst, 
            ITensorIndexer2D<float> w, int mb, int oc)
        {
            int MB = src.DimNCount;
            int OC = dst.DimZCount;
            int IC = src.DimZCount;
            int KH = w.DimYCount;
            int KW = w.DimXCount;

            var d = 0f;
            for (int ic = 0; ic < IC; ++ic)
            {
                for (int kh = 0; kh < KH; ++kh)
                {
                    for (int kw = 0; kw < KW; ++kw)
                    {
                        d += src.At(kw, kh, ic, mb)
                            * w.At(kw, kh);

                        //d += src[src_d.off(mb, ic, kh, kw)]
                        //    * weights[weights_d.off(oc, ic, kh, kw)];
                    }
                }
            }

            return d;
        }
    }
}
