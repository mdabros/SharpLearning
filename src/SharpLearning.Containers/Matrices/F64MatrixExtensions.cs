using System;
using SharpLearning.Containers.Extensions;

namespace SharpLearning.Containers.Matrices
{
    /// <summary>
    /// Extensions for F64Matrix
    /// </summary>
    public static class F64MatrixExtensions
    {
        /// <summary>
        /// Clears the matrix by setting all elements to 0.0
        /// </summary>
        /// <param name="matrix"></param>
        public static void Clear(this F64Matrix matrix)
        {
            var data = matrix.Data();
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = 0.0;
            }
        }

        /// <summary>
        /// Iterates over all elements in the matrix and applies the function to the elements.
        /// The values are updated directly in the Matrix.
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="func"></param>
        public static void Map(this F64Matrix matrix, Func<double> func)
        {
            matrix.Data().Map(func);
        }


        /// <summary>
        /// Iterates over all elements in the matrix and applies the function to the elements.
        /// The values are updated directly in the Matrix.
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="func"></param>
        public static void Map(this F64Matrix matrix, Func<double, double> func)
        {
            matrix.Data().Map(func);
        }

        /// <summary>
        /// Converts F64Matrix to StringMatrix
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static StringMatrix ToStringMatrix(this F64Matrix matrix)
        {
            var stringFeatures = Array.ConvertAll(matrix.Data(), FloatingPointConversion.ToString);
            return new StringMatrix(stringFeatures, matrix.RowCount, matrix.ColumnCount);
        }


        /// <summary>
        /// Combines vector1 and vector2 column-wise. Vector2 is added to the end of vector1 
        /// </summary>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <returns></returns>
        public static F64Matrix CombineCols(this double[] v1, double[] v2)
        {
            if (v1.Length != v2.Length)
            {
                throw new ArgumentException("vectors need same lengths");
            }

            var rows = v1.Length;
            var cols = 2;

            var features = new double[v1.Length + v2.Length];

            var featuresIndex = 0;
            for (int i = 0; i < v1.Length; i++)
            {
                features[featuresIndex] = v1[i];
                featuresIndex++;
                features[featuresIndex] = v2[i];
                featuresIndex++;
            }

            return new F64Matrix(features, rows, cols);
        }

        /// <summary>
        /// Combines matrix and vector column-wise. Vector is added to the end of the matrix 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public static F64Matrix CombineCols(this F64Matrix m, double[] v)
        {
            if (m.RowCount != v.Length)
            {
                throw new ArgumentException("matrix must have same number of rows as vector");
            }

            var rows = v.Length;
            var cols = m.ColumnCount + 1;

            var features = new double[rows * cols];
            var matrixArray = m.Data();

            var combineIndex = 0;
            for (int i = 0; i < rows; i++)
            {
                var matrixIndex = i * m.ColumnCount;
                Array.Copy(matrixArray, matrixIndex, features, combineIndex, m.ColumnCount);

                var otherIndex = i;
                combineIndex += m.ColumnCount;

                Array.Copy(v, otherIndex, features, combineIndex, 1);
                combineIndex += 1;
            }


            return new F64Matrix(features, rows, cols);
        }

        /// <summary>
        /// Combines vector and matrix column-wise. Matrix is added to the left of the vector 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public static F64Matrix CombineCols(this double[] v, F64Matrix m)
        {
            if (m.RowCount != v.Length)
            {
                throw new ArgumentException("matrix must have same number of rows as vector");
            }

            var rows = v.Length;
            var cols = m.ColumnCount + 1;

            var features = new double[rows * cols];
            var matrixArray = m.Data();

            var combineIndex = 0;
            for (int i = 0; i < rows; i++)
            {
                Array.Copy(v, i, features, combineIndex, 1);
                combineIndex += 1;
                
                var matrixIndex = i * m.ColumnCount;
                Array.Copy(matrixArray, matrixIndex, features, combineIndex, m.ColumnCount);
                combineIndex += m.ColumnCount;

            }
            
            return new F64Matrix(features, rows, cols);
        }


        /// <summary>
        /// Combines matrix1 and matrix2 column-wise. Matrix2 is added to the end of matrix1
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static F64Matrix CombineCols(this F64Matrix m1, F64Matrix m2)
        {
            if (m1.RowCount != m2.RowCount)
            {
                throw new ArgumentException("matrices must have same number of rows in order to be combined");
            }

            var rows = m1.RowCount;
            var columns = m1.ColumnCount + m2.ColumnCount;

            var matrixArray = m1.Data();
            var otherArray = m2.Data();

            var features = new double[matrixArray.Length + otherArray.Length];

            var combineIndex = 0;
            for (int i = 0; i < rows; i++)
            {
                var matrixIndex = i * m1.ColumnCount;
                Array.Copy(matrixArray, matrixIndex, features, combineIndex, m1.ColumnCount);

                var otherIndex = i * m2.ColumnCount;
                combineIndex += m1.ColumnCount;

                Array.Copy(otherArray, otherIndex, features, combineIndex, m2.ColumnCount);
                combineIndex += m2.ColumnCount;
            }


            return new F64Matrix(features, rows, columns);
        }

        /// <summary>
        /// Combines vector1 and vector2 row wise. Vector2 is added to the bottom of vector1 
        /// </summary>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <returns></returns>
        public static F64Matrix CombineRows(this double[] v1, double[] v2)
        {
            if (v1.Length != v2.Length)
            {
                throw new ArgumentException("vectors need same lengths");
            }

            var rows = 2;
            var cols = v1.Length;

            var features = new double[v1.Length + v2.Length];

            Array.Copy(v1, 0, features, 0, v1.Length);
            Array.Copy(v2, 0, features, v2.Length, v2.Length);

            return new F64Matrix(features, rows, cols);
        }


        /// <summary>
        /// Combines matrix and vector row wise. Vector is added to the bottom of the matrix 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public static F64Matrix CombineRows(this F64Matrix m, double[] v)
        {
            if (m.ColumnCount != v.Length)
            {
                throw new ArgumentException("matrix must have same number of columns as vector");
            }

            var rows = m.RowCount + 1;
            var cols = v.Length;

            var features = new double[rows * cols];
            var matrixArray = m.Data();

            Array.Copy(matrixArray, 0, features, 0, matrixArray.Length);
            Array.Copy(v, 0, features, matrixArray.Length, v.Length);

            return new F64Matrix(features, rows, cols);
        }

        /// <summary>
        /// Combines vecor and matrix row wise. Matrix is added to the bottom of the vector
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public static F64Matrix CombineRows(this double[] v, F64Matrix m)
        {
            if (m.ColumnCount != v.Length)
            {
                throw new ArgumentException("matrix must have same number of columns as vector");
            }

            var rows = m.RowCount + 1;
            var cols = v.Length;

            var features = new double[rows * cols];
            var matrixArray = m.Data();

            Array.Copy(v, 0, features, 0, v.Length);
            Array.Copy(matrixArray, 0, features, v.Length, matrixArray.Length);

            return new F64Matrix(features, rows, cols);
        }


        /// <summary>
        /// Combines matrix1 and matrix2 row wise. Matrix2 is added to the bottom of matrix1
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static F64Matrix CombineRows(this F64Matrix m1, F64Matrix m2)
        {
            if (m1.ColumnCount != m2.ColumnCount)
            {
                throw new ArgumentException("matrices must have same number of columns in order to be combined");
            }

            var rows = m1.RowCount + m2.RowCount;
            var columns = m1.ColumnCount;

            var matrixArray = m1.Data();
            var otherArray = m2.Data();

            var features = new double[matrixArray.Length + otherArray.Length];

            Array.Copy(matrixArray, 0, features, 0, matrixArray.Length);
            Array.Copy(otherArray, 0, features, matrixArray.Length, otherArray.Length);

            return new F64Matrix(features, rows, columns);
        }
    }
}
