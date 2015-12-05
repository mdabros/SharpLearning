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
        public static void AddF64(F64Matrix m, double[] v, F64Matrix output)
        {
            var rows = m.GetNumberOfRows();
            var cols = m.GetNumberOfColumns();

            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    output[j, i] = m[j, i] + v[j];
                }
            }
        }
    }
}
