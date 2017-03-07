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
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="tOut"></param>
        public static void Multiply_MathNet(Tensor<float> t1, Tensor<float> t2, Tensor<float> tOut)
        {
            if (t1.NumberOfDimensions != 2 || t2.NumberOfDimensions != 2 || tOut.NumberOfDimensions != 2)
            { throw new ArgumentException($"Only 2-dim tensors is supported"); }

            if (t1.Dimensions[1] != t2.Dimensions[0])
            { throw new ArgumentException($"matrix a cols: differs from matrix b rows: "); }


            if (tOut.Dimensions[0] != t1.Dimensions[0])
            {
                throw new ArgumentException($"output matrix rows: differs from matrix a rows: " );
            }

            if (tOut.Dimensions[1] != t2.Dimensions[1])
            {
                throw new ArgumentException($"output matrix rows: differs from matrix b cols: ");
            }

            var m1 = MathNet.Numerics.LinearAlgebra.Matrix<float>.Build.Dense(t1.Dimensions[0], t1.Dimensions[1], t1.Data);
            var m2 = MathNet.Numerics.LinearAlgebra.Matrix<float>.Build.Dense(t2.Dimensions[0], t2.Dimensions[1], t2.Data);
            var mOut = MathNet.Numerics.LinearAlgebra.Matrix<float>.Build.Dense(tOut.Dimensions[0], tOut.Dimensions[1], tOut.Data);

            m1.Multiply(m2, mOut);
        }



        /// <summary>
        /// 
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <param name="output"></param>
        public static void Multiply(Tensor<float> m1, Tensor<float> m2, Tensor<float> output)
        {
            if(m1.NumberOfDimensions != 2 || m2.NumberOfDimensions != 2 || output.NumberOfDimensions != 2)
            { throw new ArgumentException("Only 2-dimensional tensors are supported. "); }

            var m1data = m1.Data;
            var m2daa = m2.Data;
            var outData = output.Data;

            var m1H = m1.Dimensions[0];
            var m1W = m1.Dimensions[1];

            var m2H = m2.Dimensions[0];
            var m2W = m2.Dimensions[1];

            var outH = output.Dimensions[0];
            var outW = output.Dimensions[1];

            if (m1W != m2H)
            { throw new ArgumentException($"matrix a cols: {m1W} differs from matrix b rows: {m2H}"); }


            if (outH != m1H)
            {
                throw new ArgumentException($"output matrix rows: {outH} differs from matrix a rows: " + m1H);
            }

            if (outW != m2W)
            {
                throw new ArgumentException($"output matrix rows: {outW} differs from matrix b cols: {m2W}");
            }

            Parallel.For(0, m1H, i =>
            {
                var m1OffSet = m1W * i;
                var outOffSet = outW * i;

                for (int k = 0; k < m2H; k++)
                {
                    var a = m1data[m1OffSet + k];
                    var m2OffSet = m2W * k;

                    for (int j = 0; j < m2W; j++)
                    {
                        outData[outOffSet + j] += a * m2daa[m2OffSet + j];
                    }
                }
            });
        }
        
        static void InnerLoop(float[] v, float value, float[] output)
        {
            for (int i = 0; i < v.Length; i++)
            {
                output[i] = value * v[i];
            }
        }


        static void InnerLoopSimd(float[] v, float value, float[] output)
        {
            var simdLength = Vector<float>.Count;

            var i = 0;
            for (i = 0; i <= v.Length - simdLength; i += simdLength)
            {
                var va = new Vector<float>(v, i);
                var vb = new Vector<float>(value);
                var vr = va * vb;

                var vOut = new Vector<float>(output, i);
                (vr + vOut).CopyTo(output, i);
            }

            for (; i < v.Length; ++i)
            {
                output[i] = value * v[i];
            }
        }
    }
}
