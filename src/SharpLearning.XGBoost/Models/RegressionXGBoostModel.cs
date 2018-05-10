using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using XGBoost.lib;

namespace SharpLearning.XGBoost.Models
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class RegressionXGBoostModel : IDisposable, IPredictorModel<double>
    {
        readonly Booster m_model;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="model"></param>
        public RegressionXGBoostModel(Booster model)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            m_model = model;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var floatObservation = new float[][]
            {
                observation.ToFloat()
            };

            using (var data = new DMatrix(floatObservation))
            {
                return (double)m_model.Predict(data).Single();
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var floatObservations = observations.ToFloatJaggedArray();
            using (var data = new DMatrix(floatObservations))
            {
                return m_model.Predict(data).ToDouble();
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public double[] GetRawVariableImportance()
        {
            throw new System.NotImplementedException();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            throw new System.NotImplementedException();
        }

        /// <summary>
        /// Loads a RegressionXGBoostModel.
        /// </summary>
        /// <param name="modelFilePath"></param>
        /// <returns></returns>
        public static RegressionXGBoostModel Load(string modelFilePath)
        {
            // load XGBoost model.
            return new RegressionXGBoostModel(new Booster(modelFilePath));
        }

        /// <summary>
        /// Saves the RegressionXGBoostModel.
        /// </summary>
        /// <param name="modelFilePath"></param>
        public void Save(string modelFilePath)
        {
            // Save XGBoost model.
            m_model.Save(modelFilePath);
        }

        /// <summary>
        /// 
        /// </summary>
        public void Dispose()
        {
            if (m_model != null)
            {
                m_model.Dispose();
            }
        }
    }
}
