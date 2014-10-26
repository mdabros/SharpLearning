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
