using MathNet.Numerics.LinearAlgebra;
using System;

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

            if (v.Count != cols)
            { throw new ArgumentException("matrix cols: " + cols + " differs from vector length: " + v.Count); }

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    output[i, j] = m[i, j] + v[j];
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

            if (v.Count != rows)
            { throw new ArgumentException("matrix rows: " + rows + " differs from vector length: " + v.Count); }

            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    output[j, i] = m[j, i] + v[j];
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

            if (v.Count != cols)
            { throw new ArgumentException("matrix cols: " + cols + " differs from vector length: " + v.Count); }

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    output[i, j] = m[i, j] * v[j];
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
            { throw new ArgumentException("m1 rows: " + rows + " differs from m2 rows: " + m2.RowCount); }
            if (m2.ColumnCount!= cols)
            { throw new ArgumentException("m1 cols: " + cols + " differs from m2 cols: " + m2.ColumnCount); }

            var sum = 0.0f;
            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    sum += m1[j, i] * m2[j, i];
                }
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
            for (int i = 0; i < m.ColumnCount; i++)
            {
                for (int j = 0; j < m.RowCount; j++)
                {
                    v[i] += m[j, i];
                }

                v[i] = v[i] / (float)m.RowCount;
            }
        }

        /// <summary>
        /// Sums the columns of m into the vector sums. 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="sums"></param>
        public static void SumColumns(this Matrix<float> m, Vector<float> sums)
        {
            for (int i = 0; i < m.ColumnCount; i++)
            {
                for (int j = 0; j < m.RowCount; j++)
                {
                    sums[i] += m.At(j, i);
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
            for (int i = 0; i < m.RowCount; i++)
            {
                for (int j = 0; j < m.ColumnCount; j++)
                {
                    sums[i] += m.At(i, j);
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
                matrix[0, i] = (float)array[i];
            }

            return matrix;
        }
    }
}
