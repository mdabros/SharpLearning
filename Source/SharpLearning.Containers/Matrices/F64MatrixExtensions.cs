using System;

namespace SharpLearning.Containers.Matrices
{
    public static class F64MatrixExtensions
    {
        /// <summary>
        /// Combines vector1 and vector2 columnwise. Vector2 is added to the end of vector1 
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
        /// Combines matrix and vector columnwise. Vector is added to the end of the matrix 
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public static F64Matrix CombineCols(this F64Matrix m, double[] v)
        {
            if (m.GetNumberOfRows() != v.Length)
            {
                throw new ArgumentException("matrix must have same number of rows as vector");
            }

            var rows = v.Length;
            var cols = m.GetNumberOfColumns() + 1;

            var features = new double[rows * cols];
            var matrixArray = m.GetFeatureArray();

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


            return new F64Matrix(features, rows, cols);
        }

        /// <summary>
        /// Combines matrix1 and matrix2 columnwise. Matrix2 is added to the end of matrix1
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static F64Matrix CombineCols(this F64Matrix m1, F64Matrix m2)
        {
            if (m1.GetNumberOfRows() != m2.GetNumberOfRows())
            {
                throw new ArgumentException("matrices must have same number of rows inorder to be combined");
            }

            var rows = m1.GetNumberOfRows();
            var columns = m1.GetNumberOfColumns() + m2.GetNumberOfColumns();

            var matrixArray = m1.GetFeatureArray();
            var otherArray = m2.GetFeatureArray();

            var features = new double[matrixArray.Length + otherArray.Length];

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


            return new F64Matrix(features, rows, columns);
        }
    }
}
