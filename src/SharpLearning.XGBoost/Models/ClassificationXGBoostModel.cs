using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Serialization;
using XGBoost.lib;

namespace SharpLearning.XGBoost.Models
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class ClassificationXGBoostModel : IDisposable, IPredictorModel<double>
    {
        readonly Booster m_model;
        readonly Dictionary<float, double> m_targetIndexToTargetName;

        const string m_targetIndexToTargetNameFileName = "targetIndexToTargetName.xml";

        /// <summary>
        /// 
        /// </summary>
        /// <param name="model"></param>
        /// <param name="targetIndexToTargetName"></param>
        public ClassificationXGBoostModel(Booster model, Dictionary<float, double> targetIndexToTargetName)
        {
            if (model == null) throw new ArgumentNullException(nameof(model));
            if (targetIndexToTargetName == null) throw new ArgumentNullException(nameof(targetIndexToTargetName));
            m_model = model;
            m_targetIndexToTargetName = targetIndexToTargetName;
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
                return m_targetIndexToTargetName[m_model.Predict(data).Single()];
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
                return m_model.Predict(data)
                    .Select(v => m_targetIndexToTargetName[v])
                    .ToArray();
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
        /// Loads a ClassificationXGBoostModel.
        /// </summary>
        /// <param name="modelFilePath"></param>
        /// <returns></returns>
        public static ClassificationXGBoostModel Load(string modelFilePath)
        {
            // Load target mapping.
            var targetIndexToTargetNameFilePath = GetTargetIndexToTargetNameFilePath(modelFilePath);
            var targetIndexToTargetName = new GenericXmlDataContractSerializer()
                .Deserialize<Dictionary<float, double>>(() => new StreamReader(targetIndexToTargetNameFilePath));

            // load XGBoost model.
            return new ClassificationXGBoostModel(new Booster(modelFilePath), targetIndexToTargetName);
        }

        /// <summary>
        /// Saves the ClassificationXGBoostModel.
        /// </summary>
        /// <param name="modelFilePath"></param>
        public void Save(string modelFilePath)
        {
            // Save XGBoost model.
            m_model.Save(modelFilePath);

            // Save target mapping.
            var targetIndexToTargetNameFilePath = GetTargetIndexToTargetNameFilePath(modelFilePath);

            new GenericXmlDataContractSerializer()
                .Serialize(m_targetIndexToTargetName, 
                    () => new StreamWriter(targetIndexToTargetNameFilePath));
        }

        static string GetTargetIndexToTargetNameFilePath(string modelFilePath)
        {
            return Path.Combine(Path.GetDirectoryName(modelFilePath),
                m_targetIndexToTargetNameFileName);
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
