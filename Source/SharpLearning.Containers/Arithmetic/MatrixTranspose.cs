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
            var transpose = new F64Matrix(matrix.GetNumberOfColumns(), matrix.GetNumberOfRows());
            TransposeF64(matrix, transpose);

            return transpose;
        }

        /// <summary>
        /// Transposes matrix. 
        /// Output is saved in the provided matrix transposed.
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static void TransposeF64(F64Matrix matrix, F64Matrix transposed)
        {
            cachetranpose(0, matrix.GetNumberOfRows(), 0, matrix.GetNumberOfColumns(), 
                matrix.GetFeatureArray(), transposed.GetFeatureArray());
        }

        /// <summary>
        /// Cache oblivious matrix transpose from stack overflow:
        /// http://stackoverflow.com/questions/5200338/a-cache-efficient-matrix-transpose-program
        /// </summary>
        /// <param name="rb"></param>
        /// <param name="re"></param>
        /// <param name="cb"></param>
        /// <param name="ce"></param>
        /// <param name="m"></param>
        /// <param name="T"></param>
        static void cachetranpose(int rb, int re, int cb, int ce, double[] m, double[] T)
        {
            int r = re - rb, c = ce - cb;
            if (r <= 16 && c <= 16)
            {
                for (int i = rb; i < re; i++)
                {
                    for (int j = cb; j < ce; j++)
                    {
                        T[j * re + i] = m[i * ce + j];
                    }
                }
            }
            else if (r >= c)
            {
                cachetranpose(rb, rb + (r / 2), cb, ce, m, T);
                cachetranpose(rb + (r / 2), re, cb, ce, m, T);
            }
            else
            {
                cachetranpose(rb, re, cb, cb + (c / 2), m, T);
                cachetranpose(rb, re, cb + (c / 2), ce, m, T);
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
