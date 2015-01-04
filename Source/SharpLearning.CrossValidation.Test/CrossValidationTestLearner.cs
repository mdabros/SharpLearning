using SharpLearning.Containers.Matrices;
using SharpLearning.Common.Interfaces;
using System;
using System.Linq;

namespace SharpLearning.CrossValidation.Test
{
    internal class CrossValidationTestLearner : IIndexedLearner<double>
    {
        readonly int[] m_allIndices;

        public CrossValidationTestLearner(int[] allIndices)
        {
            if (allIndices == null) { throw new ArgumentNullException("allIndices"); }
            m_allIndices = allIndices;
        }

        public IPredictor<double> Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            double[] holdOut = m_allIndices.Except(indices)
                .Select(v => (double)v).ToArray();

            return new CrossValidationTestModel(holdOut);
        }
    }
}
