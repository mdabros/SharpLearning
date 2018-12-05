using System;
using System.Collections.Generic;
using System.Numerics;
using MathNet.Numerics;
using MathNet.Numerics.Providers.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;

namespace SharpLearning.Neural.Test.RefactorBranStorm
{
    public static class TensorArithmetic
    {

        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor"></param>
        /// <param name="func"></param>
        public static void Map<T>(this Tensor<T> tensor, Func<T> func)
        {
            tensor.Data.Map(func);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="func"></param>
        public static void Map<T>(this Tensor<T> tensor, Func<T, T> func)
        {
            tensor.Data.Map(func);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="output"></param>
        public static void Subtract(this Tensor<float> t1, Tensor<float> t2, Tensor<float> output)
        {
            if (t1.ElementCount != t2.ElementCount)
            { throw new ArgumentException($"t1 element count: {t1.ElementCount} differs from t2: {t2.ElementCount}"); }

            var t1Data = t1.Data;
            var t2Data = t2.Data;
            var tOutData = output.Data;

            for (int i = 0; i < t1.ElementCount; i++)
            {
                tOutData[i] = t1Data[i] - t2Data[i];
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="output"></param>
        public static void Subtract(this Tensor<double> t1, Tensor<double> t2, Tensor<double> output)
        {
            if (t1.ElementCount != t2.ElementCount)
            { throw new ArgumentException($"t1 element count: {t1.ElementCount} differs from t2: {t2.ElementCount}"); }

            var t1Data = t1.Data;
            var t2Data = t2.Data;
            var tOutData = output.Data;

            for (int i = 0; i < t1.ElementCount; i++)
            {
                tOutData[i] = t1Data[i] - t2Data[i];
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="output"></param>
        public static void PointwiseMultiply(this Tensor<float> t1, Tensor<float> t2, Tensor<float> output)
        {
            if (t1.ElementCount != t2.ElementCount)
            { throw new ArgumentException($"t1 element count: {t1.ElementCount} differs from t2: {t2.ElementCount}"); }

            var t1Data = t1.Data;
            var t2Data = t2.Data;
            var tOutData = output.Data;

            if (Vector.IsHardwareAccelerated)
            {
                var simdLength = Vector<float>.Count;
                var i = 0;
                for (i = 0; i <= t1Data.Length - simdLength; i += simdLength)
                {
                    var va = new Vector<float>(t1Data, i);
                    var vb = new Vector<float>(t2Data, i);
                    (va * vb).CopyTo(tOutData, i);
                }

                for (; i < t1Data.Length; ++i)
                {
                    tOutData[i] = t1Data[i] * t2Data[i];
                }
            }
            else
            {
                for (int i = 0; i < t1.ElementCount; i++)
                {
                    tOutData[i] = t1Data[i] * t2Data[i];
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <param name="output"></param>
        public static void PointwiseMultiply(this Tensor<double> t1, Tensor<double> t2, Tensor<double> output)
        {
            if (t1.ElementCount != t2.ElementCount)
            { throw new ArgumentException($"t1 element count: {t1.ElementCount} differs from t2: {t2.ElementCount}"); }

            var t1Data = t1.Data;
            var t2Data = t2.Data;
            var tOutData = output.Data;

            if (Vector.IsHardwareAccelerated)
            {
                var simdLength = Vector<double>.Count;
                var i = 0;
                for (i = 0; i <= t1Data.Length - simdLength; i += simdLength)
                {
                    var va = new Vector<double>(t1Data, i);
                    var vb = new Vector<double>(t2Data, i);
                    (va * vb).CopyTo(tOutData, i);
                }

                for (; i < t1Data.Length; ++i)
                {
                    tOutData[i] = t1Data[i] * t2Data[i];
                }
            }
            else
            {
                for (int i = 0; i < t1.ElementCount; i++)
                {
                    tOutData[i] = t1Data[i] * t2Data[i];
                }
            }
        }

        /// <summary>
        /// Adds vector v to matrix m. V is Added to each row of the matrix
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <param name="output"></param>
        public static void AddRowWise(this Tensor<float> m, float[] v, Tensor<float> output)
        {
            var rows = m.Dimensions[0];
            // assume tensor 2D, flatten if not
            var cols = m.DimensionOffSets[0];

            var mData = m.Data;
            var outData = output.Data;
            var vData = v;

            if (v.Length != cols)
            { throw new ArgumentException("matrix cols: " + cols + " differs from vector length: " + v.Length); }

            for (int row = 0; row < rows; row++)
            {
                var rowOffSet = cols * row;
                for (int col = 0; col < cols; col++)
                {
                    var mIndex = rowOffSet + col;
                    outData[mIndex] = mData[mIndex] + vData[col];
                }
            }
        }

        /// <summary>
        /// Adds vector v to matrix m. V is Added to each row of the matrix
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <param name="output"></param>
        public static void AddRowWise(this Tensor<double> m, double[] v, Tensor<double> output)
        {
            var rows = m.Dimensions[0];
            // assume tensor 2D, flatten if not
            var cols = m.DimensionOffSets[0];

            var mData = m.Data;
            var outData = output.Data;
            var vData = v;

            if (v.Length != cols)
            { throw new ArgumentException("matrix cols: " + cols + " differs from vector length: " + v.Length); }

            for (int row = 0; row < rows; row++)
            {
                var rowOffSet = cols * row;
                for (int col = 0; col < cols; col++)
                {
                    var mIndex = rowOffSet + col;
                    outData[mIndex] = mData[mIndex] + vData[col];
                }
            }
        }

        /// <summary>
        /// Adds vector v to matrix m. V is Added to each column of the matrix
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <param name="output"></param>
        public static void AddColumnWise(this Tensor<float> m, float[] v, Tensor<float> output)
        {
            var rows = m.Dimensions[0];
            // assume tensor 2D, flatten if not
            var cols = m.DimensionOffSets[0];

            var mData = m.Data;
            var outData = output.Data;
            var vData = v;

            if (v.Length != rows)
            { throw new ArgumentException("matrix rows: " + rows + " differs from vector length: " + v.Length); }


            for (int col = 0; col < cols; col++)
            {
                for (int row = 0; row < rows; row++)
                {
                    var rowOffSet = row * cols;
                    var mIndex = rowOffSet + col;
                    outData[mIndex] = mData[mIndex] + vData[row];
                }
            }
        }


        /// <summary>
        /// Adds vector v to matrix m. V is Added to each column of the matrix
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <param name="output"></param>
        public static void AddColumnWise(this Tensor<double> m, double[] v, Tensor<double> output)
        {
            var rows = m.Dimensions[0];
            // assume tensor 2D, flatten if not
            var cols = m.DimensionOffSets[0];

            var mData = m.Data;
            var outData = output.Data;
            var vData = v;

            if (v.Length != rows)
            { throw new ArgumentException("matrix rows: " + rows + " differs from vector length: " + v.Length); }


            for (int col = 0; col < cols; col++)
            {
                for (int row = 0; row < rows; row++)
                {
                    var rowOffSet = row * cols;
                    var mIndex = rowOffSet + col;
                    outData[mIndex] = mData[mIndex] + vData[row];
                }
            }
        }

        /// <summary>
        /// Sums the columns of m into the vector sums. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="sums"></param>
        public static void SumColumns(this Tensor<float> m, float[] sums)
        {
            var rows = m.Dimensions[0];
            // assume tensor 2D, flatten if not
            var cols = m.DimensionOffSets[0];
            var mData = m.Data;

            for (int col = 0; col < cols; col++)
            {
                for (int row = 0; row < rows; row++)
                {
                    var rowOffSet = row * cols;
                    var mIndex = rowOffSet + col;
                    sums[col] += mData[mIndex];
                }
            }
        }


        /// <summary>
        /// Sums the columns of m into the vector sums. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="sums"></param>
        public static void SumColumns(this Tensor<double> m, double[] sums)
        {
            var rows = m.Dimensions[0];
            // assume tensor 2D, flatten if not
            var cols = m.DimensionOffSets[0];
            var mData = m.Data;

            for (int col = 0; col < cols; col++)
            {
                for (int row = 0; row < rows; row++)
                {
                    var rowOffSet = row * cols;
                    var mIndex = rowOffSet + col;
                    sums[col] += mData[mIndex];
                }
            }
        }

        /// <summary>
        /// Sums the columns of m into the vector sums. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="sums"></param>
        public static void SumRows(this Tensor<float> m, float[] sums)
        {
            var rows = m.Dimensions[0];
            // assume tensor 2D, flatten if not
            var cols = m.DimensionOffSets[0];
            var mData = m.Data;

            for (int row = 0; row < rows; row++)
            {
                var rowOffSet = row * cols;
                for (int col = 0; col < cols; col++)
                {
                    var mIndex = rowOffSet + col;
                    sums[row] += mData[mIndex];
                }
            }
        }

        /// <summary>
        /// Sums the columns of m into the vector sums. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="sums"></param>
        public static void SumRows(this Tensor<double> m, double[] sums)
        {
            var rows = m.Dimensions[0];
            // assume tensor 2D, flatten if not
            var cols = m.DimensionOffSets[0];
            var mData = m.Data;

            for (int row = 0; row < rows; row++)
            {
                var rowOffSet = row * cols;
                for (int col = 0; col < cols; col++)
                {
                    var mIndex = rowOffSet + col;
                    sums[row] += mData[mIndex];
                }
            }
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
                throw new ArgumentException($"output tensor rows: {tOutRows} differs from tensor1 rows: {t1Rows}");
            }

            if (tOutCols != t2Cols)
            {
                throw new ArgumentException($"output tensor rows: {tOutCols} differs from matrix b cols: {t2Cols} ");
            }

            // clear tOut
            tOut.Data.Clear();

            // using CSIntel.Mkl, use row-major directly.
            // CBLAS.sgemm(CBLAS.ORDER.RowMajor, CBLAS.TRANSPOSE.NoTrans, CBLAS.TRANSPOSE.NoTrans,
            //    t1Rows, t2Cols, t1Cols, 1.0f, t1.Data, t1Cols, t2.Data, t2Cols, 1.0f, ref tOutData, tOutCols);

            // Switch order and dimensions in order to switch from row-major to col-major (math.net representation).
            Control.LinearAlgebraProvider.MatrixMultiplyWithUpdate(Transpose.DontTranspose, Transpose.DontTranspose,
                1.0f, t2.Data, t2Cols, t2Rows, t1.Data, t1Cols, t1Rows, 1.0f, tOut.Data);
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
            var t2Rows = t2.Dimensions[0];
            // Assumes 2D or else collapses to 2D
            var t2Cols = t2.DimensionOffSets[0];

            var tOutRows = tOut.Dimensions[0];
            // Assumes 2D or else collapses to 2D
            var tOutCols = tOut.DimensionOffSets[0];

            // since we transpose t2, t1cols much match t2 cols
            if (t1Cols != t2Cols)
            { throw new ArgumentException($"tensor1 cols: {t1Cols} differs from tensor2 rows: {t2Cols}"); }


            if (tOutRows != t1Rows)
            {
                throw new ArgumentException($"output tensor rows: {tOutRows} differs from tensor1 rows: {t1Rows}");
            }

            // since we transpose t2 toutCols much match t2rows
            if (tOutCols != t2Rows)
            {
                throw new ArgumentException($"output tensor rows: {tOutCols} differs from matrix b cols: {t2Rows} ");
            }

            // clear tOut
            tOut.Data.Clear();

            // Switch order and dimensions in order to switch from row-major to col-major (math.net representation).
            // because of order transpose t2 and don't transpose t1.
            Control.LinearAlgebraProvider
            .MatrixMultiplyWithUpdate(Transpose.Transpose, Transpose.DontTranspose,
                1.0f, t2.Data, t2Cols, t2Rows, t1.Data, t1Cols, t1Rows, 1.0f, tOut.Data);
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
            var t1Rows = t1.Dimensions[0];
            // Assumes 2D or else collapses to 2D
            var t1Cols = t1.DimensionOffSets[0];

            var t2Rows = t2.Dimensions[0];
            // Assumes 2D or else collapses to 2D
            var t2Cols = t2.DimensionOffSets[0];

            var tOutRows = tOut.Dimensions[0];
            // Assumes 2D or else collapses to 2D
            var tOutCols = tOut.DimensionOffSets[0];

            // since we expect to transpose t1rows should match t2rows
            if (t1Rows != t2Rows)
            { throw new ArgumentException($"tensor1 cols: {t1Rows} differs from tensor2 rows: {t2Rows}"); }

            // since we expect to transpose t1OutRows should match t1Cols
            if (tOutRows != t1Cols)
            {
                throw new ArgumentException($"output tensor rows: {tOutRows} differs from tensor1 rows: {t1Cols}");
            }

            if (tOutCols != t2Cols)
            {
                throw new ArgumentException($"output tensor rows: {tOutCols} differs from matrix b cols: {t2Cols} ");
            }

            // clear tOut
            tOut.Data.Clear();

            // Switch order and dimensions in order to switch from row-major to col-major (math.net representation).
            // because of order transpose t1 and don't transpose t2.
            Control.LinearAlgebraProvider
            .MatrixMultiplyWithUpdate(Transpose.DontTranspose, Transpose.Transpose,
                1.0f, t2.Data, t2Cols, t2Rows, t1.Data, t1Cols, t1Rows, 1.0f, tOut.Data);
        }
    }
}
