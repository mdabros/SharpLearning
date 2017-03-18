using System;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.Providers.LinearAlgebra;
using SharpLearning.Containers.Tensors;
using SharpLearning.Containers.Extensions;


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
                result += v1[i] * v2[i];
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
        public static void Multiply(this Tensor<float> t1, Tensor<float> t2, Tensor<float> tOut)
        {
            var t1Rows = t1.Dimensions[0];
            // Assumes 2D or else collapses to 2D
            var t1Cols = t1.DimensionOffSets[0];

            var t2Rows = t2.Dimensions[0];
            // Assumes 2D or else collapses to 2D
            var t2Cols = t2.DimensionOffSets[0];

            var tOutRows = tOut.Dimensions[0];
            // Assumes 2D or else collapses to 2D
            var tOutCols = tOut.DimensionOffSets[0];

            if (t1Cols != t2Rows)
            { throw new ArgumentException($"tensor1 cols: {t1Cols} differs from tensor2 rows: {t2Rows}"); }

            if (tOutRows != t1Rows)
            {
                throw new ArgumentException($"output tensor rows: {tOutRows} differs from tensor1 rows: {t1Rows}" );
            }

            if (tOutCols != t2Cols)
            {
                throw new ArgumentException($"output tensor rows: {tOutCols} differs from matrix b cols: {t2Cols} ");
            }

            // clear tOut
            tOut.Data.Clear();

            // transpose and switch dimensions inorder to switch from row-major to col-major (math.net representation).
            Control.LinearAlgebraProvider
            .MatrixMultiplyWithUpdate(Transpose.Transpose, Transpose.Transpose,
                1.0f, t1.Data, t1Cols, t1Rows, t2.Data, t2Cols, t2Rows, 1.0f, tOut.Data);

            // results has to be transposed inorder to get back to row-major.
            var tr1 = MathNet.Numerics.LinearAlgebra.Matrix<float>.Build.Dense(tOutRows, tOutCols, tOut.Data.ToArray());
            var tr2 = MathNet.Numerics.LinearAlgebra.Matrix<float>.Build.Dense(tOutCols, tOutRows, tOut.Data);
            tr1.Transpose(tr2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="tOut"></param>
        public static void TransposeAndMultiply(this Tensor<float> t1, Tensor<float> t2, Tensor<float> tOut)
        {
            var t1Rows = t1.Dimensions[0];
            // Assumes 2D or else collapses to 2D
            var t1Cols = t1.DimensionOffSets[0];

            // rows and cols are switched since these has to match the transposed dimensions.
            var t2Rows = t2.DimensionOffSets[0]; 
            // Assumes 2D or else collapses to 2D
            var t2Cols = t2.Dimensions[0];

            var tOutRows = tOut.Dimensions[0];
            // Assumes 2D or else collapses to 2D
            var tOutCols = tOut.DimensionOffSets[0];

            if (t1Cols != t2Rows)
            { throw new ArgumentException($"tensor1 cols: {t1Cols} differs from tensor2 rows: {t2Rows}"); }

            if (tOutRows != t1Rows)
            {
                throw new ArgumentException($"output tensor rows: {tOutRows} differs from tensor1 rows: {t1Rows}");
            }

            if (tOutCols != t2Cols)
            {
                throw new ArgumentException($"output tensor rows: {tOutCols} differs from matrix b cols: {t2Cols} ");
            }

            // clear tOut
            tOut.Data.Clear();

            // t2 should be transposed in this multiplication. Because of row-major to col-major (math.net representation),
            // t2 is kept as original and t1 transposed
            Control.LinearAlgebraProvider
            .MatrixMultiplyWithUpdate(Transpose.Transpose, Transpose.DontTranspose,
                1.0f, t1.Data, t1Cols, t1Rows, t2.Data, t2Rows, t2Cols, 1.0f, tOut.Data);

            // results has to be transposed inorder to get back to row-major.
            var tr1 = MathNet.Numerics.LinearAlgebra.Matrix<float>.Build.Dense(tOutRows, tOutCols, tOut.Data.ToArray());
            var tr2 = MathNet.Numerics.LinearAlgebra.Matrix<float>.Build.Dense(tOutCols, tOutRows, tOut.Data);
            tr1.Transpose(tr2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="tOut"></param>
        public static void TransposeThisAndMultiply(this Tensor<float> t1, Tensor<float> t2, Tensor<float> tOut)
        {
            // rows and cols are switched since these has to match the transposed dimensions.
            var t1Rows = t1.DimensionOffSets[0];
            // Assumes 2D or else collapses to 2D
            var t1Cols = t1.Dimensions[0]; 

            var t2Rows = t2.Dimensions[0];
            // Assumes 2D or else collapses to 2D
            var t2Cols = t2.DimensionOffSets[0];

            var tOutRows = tOut.Dimensions[0];
            // Assumes 2D or else collapses to 2D
            var tOutCols = tOut.DimensionOffSets[0];

            if (t1Cols != t2Rows)
            { throw new ArgumentException($"tensor1 cols: {t1Cols} differs from tensor2 rows: {t2Rows}"); }

            if (tOutRows != t1Rows)
            {
                throw new ArgumentException($"output tensor rows: {tOutRows} differs from tensor1 rows: {t1Rows}");
            }

            if (tOutCols != t2Cols)
            {
                throw new ArgumentException($"output tensor rows: {tOutCols} differs from matrix b cols: {t2Cols} ");
            }

            // clear tOut
            tOut.Data.Clear();

            // t1 should be transposed in this multiplication. Because of row-major to col-major (math.net representation),
            // t1 is kept as original and t1 transposed
            Control.LinearAlgebraProvider
            .MatrixMultiplyWithUpdate(Transpose.DontTranspose, Transpose.Transpose,
                1.0f, t1.Data, t1Rows, t1Cols, t2.Data, t2Cols, t2Rows, 1.0f, tOut.Data);

            // results has to be transposed inorder to get back to row-major.
            var tr1 = MathNet.Numerics.LinearAlgebra.Matrix<float>.Build.Dense(tOutRows, tOutCols, tOut.Data.ToArray());
            var tr2 = MathNet.Numerics.LinearAlgebra.Matrix<float>.Build.Dense(tOutCols, tOutRows, tOut.Data);
            tr1.Transpose(tr2);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <param name="output"></param>
        public static void MultiplyRef(Tensor<float> m1, Tensor<float> m2, Tensor<float> output)
        {
            if(m1.DimensionCount != 2 || m2.DimensionCount != 2 || output.DimensionCount != 2)
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
       
    }
}
