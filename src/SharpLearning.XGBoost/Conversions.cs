using System.Linq;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.XGBoost
{
    public static class Conversions
    {
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

        public static float[] ToFloat(this double[] array)
        {
            return array.Select(v => (float)v).ToArray();
        }

        public static double[] ToDouble(this float[] array)
        {
            return array.Select(v => (double)v).ToArray();
        }
    }
}
