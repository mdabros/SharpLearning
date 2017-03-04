using System.Threading.Tasks;
using SharpLearning.Containers.Tensors;
using SharpLearning.Containers.Views;

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
            ITensorIndexer2D<float> src;
            if (input.NumberOfDimensions == 4)
            {
                var src4d = input.AsTensor4D();
                src = input.AsTensor2D(src4d.N, src4d.C * src4d.H * src4d.W);
            }
            else
            {
                src = input.AsTensor2D();
            }

            var dst = output.AsTensor2D();

            var w = weights.AsTensor2D();
            var b = bias.AsTensor1D();

            int MB = dst.H;
            int OC = dst.W;
            int IC = src.W;

            var interval = Interval1D.Create(0, IC);

            Parallel.For(0, MB, mb =>
            {
                var srcValues = new float[IC];
                var wValues = new float[IC];
                src.RangeW(mb, interval, srcValues);

                for (int oc = 0; oc < OC; ++oc)
                {
                    w.RangeW(oc, interval, wValues);
                    var d = Utils.Dot(srcValues, wValues);

                    d += b.At(oc);
                    dst.At(mb, oc, d);
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
