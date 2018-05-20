using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.GradientBoost.Models;
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
        /// Raw vaiable importance is not supported by XGBoost models.
        /// </summary>
        /// <returns></returns>
        public double[] GetRawVariableImportance()
        {
            throw new System.NotImplementedException();
        }

        /// <summary>
        /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            var textTrees = m_model.DumpModelEx(fmap: string.Empty, with_stats: 1, format: "text");
            var rawVariableImportance = FeatureImportanceParser.ParseFromTreeDump(textTrees, featureNameToIndex.Count);

            var max = rawVariableImportance.Max();

            var scaledVariableImportance = rawVariableImportance
                .Select(v => (v / max) * 100.0)
                .ToArray();

            return featureNameToIndex.ToDictionary(kvp => kvp.Key, kvp => scaledVariableImportance[kvp.Value])
                        .OrderByDescending(kvp => kvp.Value)
                        .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
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
        /// Converts model to fully managed RegressionGradientBoostModel.
        /// </summary>
        /// <param name="featureCount"></param>
        /// <returns></returns>
        public RegressionGradientBoostModel ToSharpLearningGBMModel(int featureCount)
        {
            var textTrees = m_model.DumpModelEx(fmap: string.Empty, with_stats: 1, format: "text");
            var trees = XGBoostTreeConverter.FromXGBoostTextTreesToGBMTrees(textTrees);

            return new RegressionGradientBoostModel(trees, 
                learningRate: 1.0, // xgboost models assume no learning rate. 
                initialLoss: 0.5, 
                featureCount: featureCount);
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
