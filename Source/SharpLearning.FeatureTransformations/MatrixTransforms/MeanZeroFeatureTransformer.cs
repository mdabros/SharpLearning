using SharpLearning.Containers.Matrices;
using System.Collections.Generic;

namespace SharpLearning.FeatureTransformations.MatrixTransforms
{
    /// <summary>
    /// Shift data so the mean of each of the selected columns is zero
    /// </summary>
    public sealed class MeanZeroFeatureTransformer : IF64MatrixTransform
    {
        readonly Dictionary<int, double> m_featureMean;

        /// <summary>
        /// Shift data so the mean of each of the selected columns is zero
        /// </summary>
        public MeanZeroFeatureTransformer()
        {
            m_featureMean = new Dictionary<int, double>();
        }

        /// <summary>
        /// Shift data so the mean of each of the selected columns is zero
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public F64Matrix Transform(F64Matrix matrix)
        {
            var rows = matrix.GetNumberOfRows();
            var cols = matrix.GetNumberOfColumns();
            var output = new F64Matrix(rows, cols);

            Transform(matrix, output);

            return output;
        }

        /// <summary>
        /// Shift data so the mean of each of the selected columns is zero
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="output"></param>
        public void Transform(F64Matrix matrix, F64Matrix output)
        {
            var rows = matrix.GetNumberOfRows();
            var cols = matrix.GetNumberOfColumns();

            if (m_featureMean.Count == 0)
            {
                for (int i = 0; i < cols; i++)
                {
                    m_featureMean.Add(i, 0.0);
                }

                for (int i = 0; i < cols; i++)
                {
                    for (int j = 0; j < rows; j++)
                    {
                        m_featureMean[i] += matrix[j, i];
                    }
                }

                for (int i = 0; i < cols; i++)
                {
                    m_featureMean[i] /= (double)rows;
                }
            }

            for (int i = 0; i < cols; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    var value = matrix[j, i];
                    var mean = m_featureMean[i];
                    output[j, i] = value - mean;
                }
            }
        }
    }
}
