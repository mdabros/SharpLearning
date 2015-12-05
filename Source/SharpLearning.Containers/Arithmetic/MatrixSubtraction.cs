using SharpLearning.Containers.Matrices;
using System;

namespace SharpLearning.Containers.Arithmetic
{
    /// <summary>
    /// 
    /// </summary>
    public static class MatrixSubtraction
    {
        /// <summary>
        /// Subtracts vectors v1 and v2
        /// </summary>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <returns></returns>
        public static double[] SubtractF64(double[] v1, double[] v2)
        {
            if (v1.Length != v2.Length)
            {
                throw new ArgumentException(string.Format("Vectors have different lengths: v1: {0}, v2: {1}",
                    v1.Length, v2.Length));
            }
            
            var v3 = new double[v1.Length];

            for (int i = 0; i < v1.Length; i++)
            {
                v3[i] = v1[i] - v2[i];
            }

            return v3;
        }

        /// <summary>
        /// Subtracts matrix m2 from m1. Result is stored in m1.
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static void SubtractF64(F64Matrix m1, F64Matrix m2, F64Matrix output)
        {
            var cols = m1.GetNumberOfColumns();
            var rows = m1.GetNumberOfRows();

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    output[i, j] = m1[i, j] - m2[i, j];
                }
            }
        }

        /// <summary>
        /// Subtracts vectors v1 and v2
        /// </summary>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <returns></returns>
        public static double[] Subtract(this double[] v1, double[] v2)
        {
            return SubtractF64(v1, v2);
        }
    }
}
