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
        /// Adds a matrix and vector
        /// </summary>
        /// <param name="m"></param>
        /// <param name="v"></param>
        public static void AddInPlace(F64Matrix m, double[] v)
        {
            var rows = m.GetNumberOfRows();
            var cols = m.GetNumberOfColumns();

            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    m[j, i] = v[j];
                }
            }
        }
    }
}
