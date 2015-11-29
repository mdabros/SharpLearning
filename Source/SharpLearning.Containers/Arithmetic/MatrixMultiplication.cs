using SharpLearning.Containers.Matrices;
using System.Diagnostics;
using System.Threading.Tasks;

namespace SharpLearning.Containers.Arithmetic
{
    /// <summary>
    /// Contains methods for matrix multiplication
    /// </summary>
    public static class MatrixMultiplication
    {
        /// <summary>
        /// Multiply vector v with matrix a
        /// </summary>
        /// <param name="a"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public static double[] MultiplyVectorF64(F64Matrix a, double[] v)
        {
            var aRows = a.GetNumberOfRows();
            var aCols = a.GetNumberOfColumns();
            
            var aData = a.GetFeatureArray();
            var cData = new double[aRows];

            for (int i = 0; i < aRows; ++i)
            {
                for (int j = 0; j < aCols; ++j)
                {
                    cData[i] += v[j] * aData[i * aCols + j];
                }
            }
            return cData;
        }

        /// <summary>
        /// Multiply vector v with matrix a. 
        /// Copies output to provided array.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="v"></param>
        /// <param name="output"></param>
        public static void MultiplyVectorF64(F64Matrix a, double[] v, double[] output)
        {
            var aRows = a.GetNumberOfRows();
            var aCols = a.GetNumberOfColumns();

            var aData = a.GetFeatureArray();

            for (int i = 0; i < aRows; ++i)
            {
                var sum = 0.0;
                for (int j = 0; j < aCols; ++j)
                {
                    sum += v[j] * aData[i * aCols + j];
                }
                output[i] = sum;
            }
        }

        /// <summary>
        /// Multiply vector v with transposed matrix a. 
        /// The matrix is traversed like it was transposed.
        /// Copies output to provided array.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="v"></param>
        /// <param name="output"></param>
        public static void MultiplyTransposeVectorF64(F64Matrix a, double[] v, double[] output)
        {
            var aRows = a.GetNumberOfRows();
            var aCols = a.GetNumberOfColumns();

            var aData = a.GetFeatureArray();

            for (int i = 0; i < aCols; i++)
            {
                var sum = 0.0;
                for (int j = 0; j < aRows; ++j)
                {
                    sum += v[j] * a.GetItemAt(j, i);
                }
                output[i] = sum;
            }
        }


        /// <summary>
        /// Multiply vector v with scalar a
        /// </summary>
        /// <param name="v"></param>
        /// <param name="s"></param>
        /// <returns></returns>
        public static double[] MultiplyScalarF64(double[] v, double s)
        {
            var vs = new double[v.Length];

            for (int i = 0; i < v.Length; ++i)
            {
                vs[i] = v[i] * s;
            }
            return vs;
        }        

        /// <summary>
        /// Multiply vector v with scalar a
        /// </summary>
        /// <param name="v"></param>
        /// <param name="s"></param>
        /// <returns></returns>
        public static double[] Multiply(this double[] v, double s)
        {
            return MultiplyScalarF64(v, s);
        }
        
        
        /// <summary>
        /// Multiply vector v with matrix a
        /// </summary>
        /// <param name="a"></param>
        /// <param name="v"></param>
        /// <returns></returns>
        public static double[] Multiply(this F64Matrix a, double[] v)
        {
            return MultiplyVectorF64(a, v);
        }

        /// <summary>
        /// Multiply matrix a with matrix b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static F64Matrix MultiplyF64(F64Matrix a, F64Matrix b)
        {
            var aData = a.GetFeatureArray();
            var aRows = a.GetNumberOfRows();
            var aCols = a.GetNumberOfColumns();

            var bData = b.GetFeatureArray();
            var bRows = b.GetNumberOfRows();
            var bCols = b.GetNumberOfColumns();

            var cRows = aRows;
            var cCols = bCols;
            var cData = new double[cRows * cCols];

            Parallel.For(0, cRows, i =>
            {
                for (int k = 0; k < bRows; k++)
                {
                    for (int j = 0; j < cCols; j++)
                    {
                        cData[i * cCols + j] += aData[i * aCols + k] * bData[k * bCols + j];
                    }
                }
            });

            return new F64Matrix(cData, cRows, cCols);
        }

        /// <summary>
        /// Multiply matrix a with matrix b
        /// Copies output to provided matrix.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <param name="output"></param>
        public static void MultiplyF64(F64Matrix a, F64Matrix b, F64Matrix output)
        {
            var aData = a.GetFeatureArray();
            var aRows = a.GetNumberOfRows();
            var aCols = a.GetNumberOfColumns();

            var bData = b.GetFeatureArray();
            var bRows = b.GetNumberOfRows();
            var bCols = b.GetNumberOfColumns();

            var outputArray = output.GetFeatureArray();

            Parallel.For(0, aRows, i =>
            {
                for (int k = 0; k < bRows; k++)
                {
                    for (int j = 0; j < bCols; j++)
                    {
                        outputArray[i * bCols + j] += aData[i * aCols + k] * bData[k * bCols + j];
                    }
                }
            });
        }

        /// <summary>
        /// Multiply matrix aT with matrix b
        /// Matrix aT is traversed like it was transposed.
        /// Copies output to provided matrix.
        /// </summary>
        /// <param name="aT"></param>
        /// <param name="b"></param>
        /// <param name="output"></param>
        public static void MultiplyTransposeFirstF64(F64Matrix aT, F64Matrix b, F64Matrix output)
        {
            var aData = aT.GetFeatureArray();
            var aRows = aT.GetNumberOfRows();
            var aCols = aT.GetNumberOfColumns();

            var bData = b.GetFeatureArray();
            var bRows = b.GetNumberOfRows();
            var bCols = b.GetNumberOfColumns();

            var outputArray = output.GetFeatureArray();

            Parallel.For(0, aCols, i =>
            {
                for (int k = 0; k < bRows; k++)
                {
                    for (int j = 0; j < bCols; j++)
                    {
                        outputArray[i * bCols + j] += aT.GetItemAt(k, i) * bData[k * bCols + j];
                    }
                }
            });
        }


        /// <summary>
        /// Multiply matrix a with matrix b
        /// Matrix bT is traversed like it was transposed.
        /// Copies output to provided matrix.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="bT">is traversed like it was transposed</param>
        /// <param name="output"></param>
        public static void MultiplyTransposeSecondF64(F64Matrix a, F64Matrix bT, F64Matrix output)
        {
            var aData = a.GetFeatureArray();
            var aRows = a.GetNumberOfRows();
            var aCols = a.GetNumberOfColumns();

            var bData = bT.GetFeatureArray();
            var bRows = bT.GetNumberOfRows();
            var bCols = bT.GetNumberOfColumns();

            var outputArray = output.GetFeatureArray();

            Parallel.For(0, aRows, i =>
            {
                for (int k = 0; k < aCols; k++)
                {
                    for (int j = 0; j < aRows; j++)
                    {
                        outputArray[i * aRows + j] += aData[i * aCols + k] * bT.GetItemAt(j, k);
                    }
                }
            });
        }


        /// <summary>
        /// Multiply matrix a with matrix b
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static F64Matrix Multiply(this F64Matrix a, F64Matrix b)
        {
            return MultiplyF64(a, b);
        }
    }

}
