using System.Linq;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Extensions;

namespace SharpLearning.XGBoost
{
    /// <summary>
    /// Conversions for XGBoost
    /// </summary>
    public static class Conversions
    {
        /// <summary>
        /// Converts F64Matrix to float[][].
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static float[][] ToFloatJaggedArray(this F64Matrix matrix)
        {
            var rows = matrix.RowCount;
            var cols = matrix.ColumnCount;

            var jaggedArray = new float[rows][];
            for (int row = 0; row < rows; row++)
            {
                var rowArray = new float[cols];

                for (int col = 0; col < cols; col++)
                {
                    rowArray[col] = (float)matrix.At(row, col);
                }
                jaggedArray[row] = rowArray;
            }

            return jaggedArray;
        }

        /// <summary>
        /// Converts F64Matrix to float[][].
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="rowIndices"></param>
        /// <returns></returns>
        public static float[][] ToFloatJaggedArray(this F64Matrix matrix, int[] rowIndices)
        {
            var rows = rowIndices.Length;
            var cols = matrix.ColumnCount;

            var jaggedArray = new float[rows][];
            for (int outputRow = 0; outputRow < rowIndices.Length; outputRow++)
            {
                var inputRow = rowIndices[outputRow];
                var rowArray = new float[cols];
                
                for (int col = 0; col < cols; col++)
                {
                    rowArray[col] = (float)matrix.At(inputRow, col);
                }
                jaggedArray[outputRow] = rowArray;
            }

            return jaggedArray;
        }

        /// <summary>
        /// Converts double array to float array.
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public static float[] ToFloat(this double[] array)
        {
            return array.Select(v => (float)v).ToArray();
        }

        /// <summary>
        /// Converts double array to float array.
        /// </summary>
        /// <param name="array"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public static float[] ToFloat(this double[] array, int[] indices)
        {
            return indices.Select(i => (float)array[i]).ToArray();            
        }

        /// <summary>
        /// Converts float array to double array.
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public static double[] ToDouble(this float[] array)
        {
            return array.Select(v => (double)v).ToArray();
        }
    }
}
