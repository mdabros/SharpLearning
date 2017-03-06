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
                //var colInterval = Interval1D.Create(0, m2.W);
                //var outValues = new float[colInterval.Length];
                //var m2Values = new float[colInterval.Length];

                //output.RangeW(i, colInterval, outValues);
                //m2.RangeW(i, colInterval, m2Values);

                //var rowInterval = Interval1D.Create(0, m1.W);
                //var m1values = new float[rowInterval.Length];

                //m1.RangeW(i, rowInterval, m1values);

                for (int k = 0; k < m2.H; k++)
                {
                    //var m1Value = m1values[k];
                    var a = m1.At(i, k);

                    for (int j = 0; j < m2.W; j++)
                    {
                        var value = output.At(i, j);
                        value += a * m2.At(k, j);
                        output.At(i, j, value);
                    }

                    //// only works with ref arrays
                    //InnerLoop(m2Values, m1Value, outValues);
                    //InnerLoopSimd(m2Values, m1Value, outValues);
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
