using System;
using SharpLearning.Containers.Extensions;

namespace SharpLearning.Containers.Matrices
{
    /// <summary>
    /// Extension methods for StringMatrix
    /// </summary>
    public static class StringMatrixExtensions
    {
        /// <summary>
        /// Converts StringMatrix to F64Matrix. This will fail if there are values not parsable as a double
        /// </summary>
        /// <param name="stringMatrix"></param>
        /// <returns></returns>
        public static F64Matrix ToF64Matrix(this StringMatrix stringMatrix)
        {
            var features = Array.ConvertAll(stringMatrix.Data(), FloatingPointConversion.ToF64);
            return new F64Matrix(features, stringMatrix.GetNumberOfRows(), stringMatrix.GetNumberOfColumns());
        }

        /// <summary>
        /// Iterates over all elements in the matrix and applies the function to the elements.
        /// The values are updated directly in the Matrix.
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="func"></param>
        public static void Map(this StringMatrix matrix, Func<string> func)
        {
            matrix.Data().Map(func);
        }


        /// <summary>
        /// Iterates over all elements in the matrix and applies the function to the elements.
        /// The values are updated directly in the Matrix.
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="func"></param>
        public static void Map(this StringMatrix matrix, Func<string, string> func)
        {
            matrix.Data().Map(func);
        }

        /// <summary>
        /// Combines vector1 and vector2 columnwise. Vector2 is added to the end of vector1 
        /// </summary>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <returns></returns>
        public static StringMatrix CombineCols(this string[] v1, string[] v2)
        {
            if (v1.Length != v2.Length)
            {
                throw new ArgumentException("vectors need same lengths");
            }

            var rows = v1.Length;
            var cols = 2;

            var features = new string[v1.Length + v2.Length];

            var featuresIndex = 0;
            for (int i = 0; i < v1.Length; i++)
            {
                features[featuresIndex] = v1[i];
                featuresIndex++;
                features[featuresIndex] = v2[i];
                featuresIndex++;
            }

            return new StringMatrix(features, rows, cols);
        }

        /// <summary>
        /// Combines matrix and vector columnwise. Vector is added to the end of the matrix 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public static StringMatrix CombineCols(this StringMatrix m, string[] v)
        {
            if (m.GetNumberOfRows() != v.Length)
            {
                throw new ArgumentException("matrix must have same number of rows as vector");
            }

            var rows = v.Length;
            var cols = m.GetNumberOfColumns() + 1;

            var features = new string[rows * cols];
            var matrixArray = m.Data();

            var combineIndex = 0;
            for (int i = 0; i < rows; i++)
            {
                var matrixIndex = i * m.GetNumberOfColumns();
                Array.Copy(matrixArray, matrixIndex, features, combineIndex, m.GetNumberOfColumns());

                var otherIndex = i;
                combineIndex += m.GetNumberOfColumns();

                Array.Copy(v, otherIndex, features, combineIndex, 1);
                combineIndex += 1;
            }


            return new StringMatrix(features, rows, cols);
        }

        /// <summary>
        /// Combines vector and and matrix  columnwise. Vector is added to the front of the matrix 
        /// </summary>
        /// <param name="v"></param>
        /// <param name="m"></param>
        /// <returns></returns>
        public static StringMatrix CombineCols(this string[] v, StringMatrix m)
        {
            if (m.GetNumberOfRows() != v.Length)
            {
                throw new ArgumentException("matrix must have same number of rows as vector");
            }

            var rows = v.Length;
            var cols = m.GetNumberOfColumns() + 1;

            var features = new string[rows * cols];
            var matrixArray = m.Data();

            var combineIndex = 0;
            for (int i = 0; i < rows; i++)
            {
                Array.Copy(v, i, features, combineIndex, 1);
                combineIndex += 1;

                var matrixIndex = i * m.GetNumberOfColumns();
                Array.Copy(matrixArray, matrixIndex, features, combineIndex, m.GetNumberOfColumns());
                combineIndex += m.GetNumberOfColumns();
            }

            return new StringMatrix(features, rows, cols);
        }

        /// <summary>
        /// Combines matrix1 and matrix2 columnwise. Matrix2 is added to the end of matrix1 
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static StringMatrix CombineCols(this StringMatrix m1, StringMatrix m2)
        {
            if (m1.GetNumberOfRows() != m2.GetNumberOfRows())
            {
                throw new ArgumentException("matrices must have same number of rows inorder to be combined");
            }

            var rows = m1.GetNumberOfRows();
            var columns = m1.GetNumberOfColumns() + m2.GetNumberOfColumns();

            var matrixArray = m1.Data();
            var otherArray = m2.Data();

            var features = new string[matrixArray.Length + otherArray.Length];

            var combineIndex = 0;
            for (int i = 0; i < rows; i++)
            {
                var matrixIndex = i * m1.GetNumberOfColumns();
                Array.Copy(matrixArray, matrixIndex, features, combineIndex, m1.GetNumberOfColumns());

                var otherIndex = i * m2.GetNumberOfColumns();
                combineIndex += m1.GetNumberOfColumns();

                Array.Copy(otherArray, otherIndex, features, combineIndex, m2.GetNumberOfColumns());
                combineIndex += m2.GetNumberOfColumns();
            }

            return new StringMatrix(features, rows, columns);
        }

        /// <summary>
        /// Combines matrix1 and matrix2 rowwise. Matrix2 is added to the end of matrix1 
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static StringMatrix CombineRows(this StringMatrix m1, StringMatrix m2)
        {
            if (m1.GetNumberOfColumns() != m2.GetNumberOfColumns())
            {
                throw new ArgumentException("matrices must have same number of rows inorder to be combined");
            }

            var rows = m1.GetNumberOfRows() + m2.GetNumberOfRows();
            var columns = m1.GetNumberOfColumns();

            var matrixArray = m1.Data();
            var otherArray = m2.Data();

            var features = new string[matrixArray.Length + otherArray.Length];

            Array.Copy(matrixArray, features, matrixArray.Length);
            Array.Copy(otherArray, 0, features, matrixArray.Length, otherArray.Length);

            return new StringMatrix(features, rows, columns);
        }
    }
}
