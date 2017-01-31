using SharpLearning.Containers.Matrices;
using System;
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
            var rows = a.GetNumberOfRows();
            var cols = a.GetNumberOfColumns();
            
            var data = a.Data();
            var output = new double[rows];

            if (cols != v.Length) 
            { throw new ArgumentException("matrix cols: " + cols + " differs from vector length: " + v.Length); }

            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    output[i] += v[j] * data[i * cols + j];
                }
            }
            return output;
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
            var rows = a.GetNumberOfRows();
            var cols = a.GetNumberOfColumns();

            var data = a.Data();

            if (cols != v.Length)
            { throw new ArgumentException("matrix cols: " + cols + " differs from vector length: " + v.Length); }


            for (int i = 0; i < rows; ++i)
            {
                var sum = 0.0;
                for (int j = 0; j < cols; ++j)
                {
                    sum += v[j] * data[i * cols + j];
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
            var aData = a.Data();
            var aRows = a.GetNumberOfRows();
            var aCols = a.GetNumberOfColumns();

            var bData = b.Data();
            var bRows = b.GetNumberOfRows();
            var bCols = b.GetNumberOfColumns();

            var cRows = aRows;
            var cCols = bCols;
            var cData = new double[cRows * cCols];

            if (aCols != bRows)
            { throw new ArgumentException("matrix a cols: " + aCols + " differs from matrix b rows: " + bRows); }

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
            var aData = a.Data();
            var aRows = a.GetNumberOfRows();
            var aCols = a.GetNumberOfColumns();

            var bData = b.Data();
            var bRows = b.GetNumberOfRows();
            var bCols = b.GetNumberOfColumns();

            var outputArray = output.Data();

            if (aCols != bRows)
            { throw new ArgumentException("matrix a cols: " + aCols + " differs from matrix b rows: " + bRows); }


            if (output.GetNumberOfRows() != aRows)
            { throw new ArgumentException("output matrix rows: " + output.GetNumberOfRows() 
                + " differs from matrix a rows: " + aRows); }

            if (output.GetNumberOfColumns() != bCols)
            {
                throw new ArgumentException("output matrix rows: " + output.GetNumberOfColumns()
                  + " differs from matrix b cols: " + bCols);
            }

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
