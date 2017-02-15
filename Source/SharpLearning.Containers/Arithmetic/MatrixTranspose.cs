using SharpLearning.Containers.Matrices;

namespace SharpLearning.Containers.Arithmetic
{
    /// <summary>
    /// Contains methods for matrix transpose
    /// </summary>
    public static class MatrixTranspose
    {
        /// <summary>
        /// Transposes matrix
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static F64Matrix TransposeF64(F64Matrix matrix)
        {
            var transpose = new F64Matrix(matrix.ColumnCount, matrix.RowCount);
            TransposeF64(matrix, transpose);

            return transpose;
        }

        /// <summary>
        /// Transposes matrix. 
        /// Output is saved in the provided matrix transposed.
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="transposed"></param>
        /// <returns></returns>
        public static void TransposeF64(F64Matrix matrix, F64Matrix transposed)
        {
            var cols = matrix.ColumnCount;
            var rows = matrix.RowCount;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    transposed.At(j, i, matrix.At(i, j));
                }
            }
        }

        /// <summary>
        /// Transposes matrix
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static F64Matrix Transpose(this F64Matrix matrix)
        {
            return TransposeF64(matrix);
        }
    }
}
