using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using XGBoost.lib;

namespace SharpLearning.XGBoost.Models
{
    public sealed class ClassificationXGBoostModel : IDisposable, IPredictorModel<double>
    {
        readonly Booster m_model;
        readonly Dictionary<float, double> m_targetIndexToTargetName;

        public ClassificationXGBoostModel(Booster model, Dictionary<float, double> targetIndexToTargetName)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (targetIndexToTargetName == null) throw new ArgumentNullException(nameof(targetIndexToTargetName));
            m_model = model;
            m_targetIndexToTargetName = targetIndexToTargetName;
        }

        public double Predict(double[] observation)
        {
            var floatObservation = new float[][]
            {
                observation.ToFloat()
            };

            using (var data = new DMatrix(floatObservation))
            {
                return m_targetIndexToTargetName[m_model.Predict(data).Single()];
            }
        }

        public double[] Predict(F64Matrix observations)
        {
            var floatObservations = observations.ToFloatJaggedArray();
            using (var data = new DMatrix(floatObservations))
            {
                return m_model.Predict(data)
                    .Select(v => m_targetIndexToTargetName[v])
                    .ToArray();
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
