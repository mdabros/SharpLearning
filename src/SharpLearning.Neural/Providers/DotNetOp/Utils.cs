using System;
using System.Numerics;
using System.Threading.Tasks;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public static class Utils
    {
        /// <summary>
        /// SIMD multiply
        /// </summary>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <returns></returns>
        public static float DotSimd(float[] v1, float[] v2)
        {
            var simdLength = Vector<float>.Count;
            var i = 0;

            var result = 0f;

            for (i = 0; i <= v1.Length - simdLength; i += simdLength)
            {
                var va = new Vector<float>(v1, i);
                var vb = new Vector<float>(v2, i);
                result += Vector.Dot(va, vb);
            }

            for (; i < v1.Length; ++i)
            {
                result = v1[i] * v2[i];
            }

            return result;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <returns></returns>
        public static float Dot(float[] v1, float[] v2)
        {
            var result = 0f;

            for (int i = 0; i < v1.Length; i++)
            {
                result += v1[i] * v2[i];
            }

            return result;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <param name="output"></param>
        public static void Multiply(ITensorIndexer2D<float> m1, ITensorIndexer2D<float> m2, ITensorIndexer2D<float> output)
        {
            if (m1.W != m2.H)
            { throw new ArgumentException($"matrix a cols: {m1.W} differs from matrix b rows: {m2.H}"); }


            if (output.H != m1.H)
            {
                throw new ArgumentException($"output matrix rows: {output.H} differs from matrix a rows: " + m1.H);
            }

            if (output.W != m2.W)
            {
                throw new ArgumentException($"output matrix rows: {output.W} differs from matrix b cols: {m2.W}");
            }
            
            Parallel.For(0, m1.H, i =>
            {
                for (int k = 0; k < m2.H; k++)
                {
                    var a = m1.At(i, k);

                    for (int j = 0; j < m2.W; j++)
                    {
                        var value = output.At(i, j);
                        value += a * m2.At(k, j);
                        output.At(i, j, value);
                    }
                }
            });
        }
    }
}
