using SharpLearning.Learners.Interfaces;
using System;

namespace SharpLearning.CrossValidation.Test
{

    internal class CrossValidationTestModel : IPredictor<double>
    {
        readonly double[] m_data;
        int m_currentIndex;

        public CrossValidationTestModel(double[] data)
        {
            if (data == null) { throw new ArgumentNullException("data"); }
            m_data = data;
        }

        public double Predict(double[] observation)
        {
            if (m_currentIndex >= m_data.Length)
            {
                m_currentIndex = 0;
            }

            return m_data[m_currentIndex++];
        }
    }
}
