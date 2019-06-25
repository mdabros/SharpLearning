using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;

namespace SharpLearning.Neural
{
    /// <summary>
    /// Extension methods to Math.net numerics.
    /// </summary>
    public static class MathNetExtensions
    {
        /// <summary>
        /// Adds vector v to matrix m. V is Added to each row of the matrix
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <param name="output"></param>
        public static void AddRowWise(this Matrix<float> m, Vector<float> v, Matrix<float> output)
        {
            var rows = m.RowCount;
            var cols = m.ColumnCount;

            var mData = m.Data();
            var outData = output.Data();
            var vData = v.Data();

            if (v.Count != cols)
            {
                throw new ArgumentException("matrix cols: " + cols + 
                    " differs from vector length: " + v.Count);
            }

            for (int col = 0; col < cols; col++)
            {
                var rowOffSet = col * rows;
                for (int row = 0; row < rows; row++)
                {
                    var mIndex = rowOffSet + row;
                    outData[mIndex] = mData[mIndex] + vData[col];
                }
            }
        }

        /// <summary>
        /// Subtracts vector v from matrix m - rowwise.
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <param name="output"></param>
        public static void SubtractRowWise(this Matrix<float> m, Vector<float> v, Matrix<float> output)
        {
            var rows = m.RowCount;
            var cols = m.ColumnCount;

            var mData = m.Data();
            var outData = output.Data();
            var vData = v.Data();

            if (v.Count != cols)
            {
                throw new ArgumentException("matrix cols: " + cols + 
                    " differs from vector length: " + v.Count);
            }

            for (int col = 0; col < cols; col++)
            {
                var rowOffSet = col * rows;
                for (int row = 0; row < rows; row++)
                {
                    var mIndex = rowOffSet + row;
                    outData[mIndex] = mData[mIndex] - vData[col];
                }
            }
        }

        /// <summary>
        /// Adds vector v to matrix m. V is Added to each column of the matrix
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <param name="output"></param>
        public static void AddColumnWise(this Matrix<float> m, Vector<float> v, Matrix<float> output)
        {
            var rows = m.RowCount;
            var cols = m.ColumnCount;

            var mData = m.Data();
            var outData = output.Data();
            var vData = v.Data();

            if (v.Count != rows)
            {
                throw new ArgumentException("matrix rows: " + rows + 
                    " differs from vector length: " + v.Count);
            }

            for (int col = 0; col < cols; col++)
            {
                var rowOffSet = col * rows;
                for (int row = 0; row < rows; row++)
                {
                    var mIndex = rowOffSet + row;
                    outData[mIndex] = mData[mIndex] + vData[row];
                }
            }
        }

        /// <summary>
        /// Multiplies vector v to matrix m. V is multiplied to each row of the matrix
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <param name="output"></param>
        public static void Multiply(this Matrix<float> m, Vector<float> v, Matrix<float> output)
        {
            var rows = m.RowCount;
            var cols = m.ColumnCount;

            var mData = m.Data();
            var outData = output.Data();
            var vData = v.Data();

            if (v.Count != cols)
            {
                throw new ArgumentException("matrix cols: " + cols + 
                    " differs from vector length: " + v.Count);
            }

            for (int col = 0; col < cols; col++)
            {
                var rowOffSet = col * rows;
                for (int row = 0; row < rows; row++)
                {
                    var mIndex = rowOffSet + row;
                    outData[mIndex] = mData[mIndex] * v[col];
                }
            }
        }

        /// <summary>
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static float ElementWiseMultiplicationSum(this Matrix<float> m1, Matrix<float> m2)
        {
            var rows = m1.RowCount;
            var cols = m1.ColumnCount;

            if (m2.RowCount != rows)
            {
                throw new ArgumentException("m1 rows: " + rows + 
                    " differs from m2 rows: " + m2.RowCount);
            }

            if (m2.ColumnCount!= cols)
            {
                throw new ArgumentException("m1 cols: " + cols + 
                    " differs from m2 cols: " + m2.ColumnCount);
            }

            var m1Data = m1.Data();
            var m2Data = m2.Data();

            var sum = 0.0f;
            for (int i = 0; i < m1Data.Length; i++)
            {
                sum += m1Data[i] * m2Data[i];
            }

            return sum;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        public static void ColumnWiseMean(this Matrix<float> m, Vector<float> v)
        {
            var rows = m.RowCount;
            
            var mData = m.Data();
            var vData = v.Data();

            for (int col = 0; col < m.ColumnCount; col++)
            {
                var rowOffSet = col * rows;
                for (int row = 0; row < m.RowCount; row++)
                {
                    var mIndex = rowOffSet + row;
                    vData[col] += mData[mIndex];
                }

                vData[col] = vData[col] / (float)m.RowCount;
            }
        }

        /// <summary>
        /// Sums the columns of m into the vector sums. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="sums"></param>
        public static void SumColumns(this Matrix<float> m, Vector<float> sums)
        {
            SumColumns(m, sums.Data());
        }

        /// <summary>
        /// Sums the columns of m into the vector sums. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="sums"></param>
        public static void SumColumns(this Matrix<float> m, float[] sums)
        {
            var rows = m.RowCount;
            var cols = m.ColumnCount;
            var mData = m.Data();

            for (int col = 0; col < m.ColumnCount; col++)
            {
                var rowOffSet = col * rows;
                for (int row = 0; row < m.RowCount; row++)
                {
                    var mIndex = rowOffSet + row;
                    sums[col] += mData[mIndex];
                }
            }
        }

        /// <summary>
        /// Sums the rows of m into the vector sums. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="sums"></param>
        public static void SumRows(this Matrix<float> m, Vector<float> sums)
        {
            SumRows(m, sums.Data());
        }

        /// <summary>
        /// Sums the rows of m into the vector sums. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="sums"></param>
        public static void SumRows(this Matrix<float> m, float[] sums)
        {
            var rows = m.RowCount;
            var cols = m.ColumnCount;

            var mData = m.Data();

            for (int col = 0; col < m.ColumnCount; col++)
            {
                var rowOffSet = col * rows;
                for (int row = 0; row < m.RowCount; row++)
                {
                    var mIndex = rowOffSet + row;
                    sums[row] += mData[mIndex];
                }
            }
        }

        /// <summary>
        /// Converts an array to a row wise matrix
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public static Matrix<float> ConvertDoubleArray(this double[] array)
        {
            var matrix = Matrix<float>.Build.Dense(1, array.Length);

            for (int i = 0; i < array.Length; i++)
            {
                matrix.At(0, i, (float)array[i]);
            }

            return matrix;
        }

        /// <summary>
        /// Copies matrix row i into array row.
        /// </summary>
        /// <param name="m"></param>
        /// <param name="rowIndex"></param>
        /// <param name="row"></param>
        public static void Row(this Matrix<float> m, int rowIndex, float[] row)
        {
            for (int i = 0; i < m.ColumnCount; i++)
            {
                row[i] = m[rowIndex, i];
            }
        }

        /// <summary>
        /// Gets the underlying data array from the matrix. 
        /// Data is stored as Column-Major.
        /// </summary>
        /// <param name="m"></param>
        /// <returns></returns>
        public static float[] Data(this Matrix<float> m)
        {
            return ((DenseColumnMajorMatrixStorage<float>)(m.Storage)).Data;
        }

        /// <summary>
        /// Gets the underlying data array from the vector. 
        /// </summary>
        /// <param name="m"></param>
        /// <returns></returns>
        public static float[] Data(this Vector<float> m)
        {
            return ((DenseVectorStorage<float>)(m.Storage)).Data;
        }
    }
}
