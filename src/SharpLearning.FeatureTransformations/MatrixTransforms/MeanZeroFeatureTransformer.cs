using SharpLearning.Containers.Matrices;
using System.Collections.Generic;
using System;

namespace SharpLearning.FeatureTransformations.MatrixTransforms
{
    /// <summary>
    /// Shift data so the mean of each of the columns is zero
    /// </summary>
    [Serializable]
    public sealed class MeanZeroFeatureTransformer : IF64MatrixTransform, IF64VectorTransform
    {
        readonly Dictionary<int, double> m_featureMean;

        /// <summary>
        /// Shift data so the mean of each of the columns is zero
        /// </summary>
        public MeanZeroFeatureTransformer()
        {
            m_featureMean = new Dictionary<int, double>();
        }

        /// <summary>
        /// Shift data so the mean of each of the columns is zero
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        public double[] Transform(double[] vector)
        {
            var output = new double[vector.Length];
            Transform(vector, output);

            return output;
        }

        /// <summary>
        /// Shift data so the mean of each of the columns is zero
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public F64Matrix Transform(F64Matrix matrix)
        {
            var rows = matrix.RowCount;
            var cols = matrix.ColumnCount;
            var output = new F64Matrix(rows, cols);

            Transform(matrix, output);

            return output;
        }

        /// <summary>
        /// Shift data so the mean of each of the columns is zero
        /// </summary>
        /// <param name="vector"></param>
        /// <param name="output"></param>
        public void Transform(double[] vector, double[] output)
        {
            if (m_featureMean.Count == 0)
            {
                throw new ArgumentException("No feature means calculated. " +
                    "Feature means must be calculated from a full matrix before the vector transform can be used");
            }

            var cols = vector.Length;

            for (int i = 0; i < cols; i++)
            {
                var value = vector[i];
                var mean = m_featureMean[i];
                output[i] = value - mean;
            }
        }

        /// <summary>
        /// Shift data so the mean of each of the columns is zero
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="output"></param>
        public void Transform(F64Matrix matrix, F64Matrix output)
        {
            var rows = matrix.RowCount;
            var cols = matrix.ColumnCount;

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
