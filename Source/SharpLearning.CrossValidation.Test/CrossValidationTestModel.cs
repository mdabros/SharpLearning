using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.CrossValidation.Test
{
    internal class CrossValidationTestModel
    {
        double[] m_data;
        int m_currentIndex;

        public CrossValidationTestModel(double[] data)
        {
            if (data == null) { throw new ArgumentNullException("data"); }
            m_data = data;
        }

        public double Predict(double[] observation)
        {
            if (m_currentIndex >= observation.Length)
            {
                m_currentIndex = 0;
            }

            return m_data[m_currentIndex++];
        }
    }
}
