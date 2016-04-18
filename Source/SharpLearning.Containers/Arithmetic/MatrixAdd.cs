using SharpLearning.Containers.Matrices;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Containers.Arithmetic
{
    public static class MatrixAdd
    {
        /// <summary>
        /// Adds a matrix and vector and stores the result in output
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <param name="output"></param>
        public static void AddF64(F64Matrix m, double[] v, F64Matrix output)
        {    
            var rows = m.GetNumberOfRows();
            var cols = m.GetNumberOfColumns();

            if (v.Length != rows) 
            { throw new ArgumentException("matrix rows: " + rows + " differs from vector length: " + v.Length); }

            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    output[j, i] = m[j, i] + v[j];
                }
            }
        }

        /// <summary>
        /// Adds two vectos of equal lengths
        /// </summary>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <param name="output"></param>
        public static double[] AddF64(double[] v1, double[] v2)
        {
            if (v1.Length != v2.Length)
            { throw new ArgumentException("v1 length: " + v1.Length + " differs from v2 length: " + v2.Length); }

            var v3 = new double[v1.Length];

            for (int i = 0; i < v1.Length; i++)
            {
                v3[i] = v1[i] + v2[i];
            }

            return v3;
        }

        /// <summary>
        /// Adds two vectors of equal length
        /// </summary>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <returns></returns>
        public static double[] Add(this double[] v1, double[] v2)
        {
            return MatrixAdd.AddF64(v1, v2);
        }
    }
}
