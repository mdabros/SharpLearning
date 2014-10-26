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
            var tCols = matrix.GetNumberOfRows();
            var tRows = matrix.GetNumberOfColumns();
            var tData = new double[tCols * tRows];
            var transpose = new F64Matrix(tData, tRows, tCols);

            var cols = matrix.GetNumberOfColumns();
            var rows = matrix.GetNumberOfRows();

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    transpose.SetItemAt(j, i, matrix.GetItemAt(i, j));
                }
            }

            return transpose;
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
