using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using XGBoost.lib;

namespace SharpLearning.XGBoost.Models
{
    public sealed class RegressionXGBoostModel : IDisposable, IPredictorModel<double>
    {
        readonly Booster m_model;

        public RegressionXGBoostModel(Booster model)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            m_model = model;
        }

        public double Predict(double[] observation)
        {
            var floatObservation = new float[1][];
            floatObservation[0] = observation.ToFloat();
            using (var test = new DMatrix(floatObservation))
            {
                return (double)m_model.Predict(test).Single();
            }
        }

        public double[] Predict(F64Matrix observations)
        {
            var floatObservations = observations.ToFloatJaggedArray();
            using (var test = new DMatrix(floatObservations))
            {
                return m_model.Predict(test).ToDouble();
            }
        }

        public double[] GetRawVariableImportance()
        {
            throw new System.NotImplementedException();
        }

        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            throw new System.NotImplementedException();
        }

        public void Dispose()
        {
            if (m_model != null)
            {
                m_model.Dispose();
            }
        }
    }
}
