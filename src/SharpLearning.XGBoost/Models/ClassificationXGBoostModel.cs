﻿using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using XGBoost.lib;

namespace SharpLearning.XGBoost.Models
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class ClassificationXGBoostModel : IDisposable, IPredictorModel<double>, IPredictorModel<ProbabilityPrediction>
    {
        readonly Booster m_model;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="model"></param>
        /// <param name="targetIndexToTargetName"></param>
        public ClassificationXGBoostModel(Booster model)
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
                var prediction = m_model.Predict(data);

                var numberOfClasses = prediction.Length;
                if (numberOfClasses >= 2)
                {
                    return PredictMultiClass(prediction);
                }
                else
                {
                    return PredictSingleClass(prediction);
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var rows = observations.RowCount;
            var cols = observations.ColumnCount;
            var observation = new double[cols];

            var predictions = new double[rows];
            for (int row = 0; row < rows; row++)
            {
                observations.Row(row, observation);
                predictions[row] = Predict(observation);
            }

            return predictions;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            var floatObservation = new float[][]
            {
                observation.ToFloat()
            };

            using (var data = new DMatrix(floatObservation))
            {
                var prediction = m_model.Predict(data);

                var numberOfClasses = prediction.Length;
                if (numberOfClasses >= 2)
                {
                    return PredictMultiClassProbability(prediction);
                }
                else
                {
                    return PredictSingleClassProbability(prediction);
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public ProbabilityPrediction[] PredictProbability(F64Matrix observations)
        {
            var rows = observations.RowCount;
            var cols = observations.ColumnCount;
            var observation = new double[cols];

            var predictions = new ProbabilityPrediction[rows];
            for (int row = 0; row < rows; row++)
            {
                observations.Row(row, observation);
                predictions[row] = PredictProbability(observation);
            }

            return predictions;
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
            // load XGBoost model.
            return new ClassificationXGBoostModel(new Booster(modelFilePath));
        }

        /// <summary>
        /// Saves the ClassificationXGBoostModel.
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

        ProbabilityPrediction IPredictor<ProbabilityPrediction>.Predict(double[] observation)
        {
            return PredictProbability(observation);
        }

        ProbabilityPrediction[] IPredictor<ProbabilityPrediction>.Predict(F64Matrix observations)
        {
            return PredictProbability(observations);
        }

        static ProbabilityPrediction PredictMultiClassProbability(float[] prediction)
        {
            var probabilities = new Dictionary<double, double>();
            var predictedClass = 0;

            var max = float.MinValue;
            for (var classIndex = 0; classIndex < prediction.Length; classIndex++)
            {
                var probability = prediction[classIndex];
                probabilities.Add(classIndex, probability);

                if (probability > max)
                {
                    max = probability;
                    predictedClass = classIndex;
                }
            }

            return new ProbabilityPrediction(predictedClass, probabilities);
        }

        static ProbabilityPrediction PredictSingleClassProbability(float[] prediction)
        {
            var predictedClass = 0;
            var singlePrediction = prediction.Single();

            if (singlePrediction > 0.5)
            {
                predictedClass = 1;
            }

            return new ProbabilityPrediction(predictedClass,
                new Dictionary<double, double> { { 0, 1 - singlePrediction }, { 1, singlePrediction } });
        }

        static double PredictMultiClass(float[] prediction)
        {
            var predictedClass = 0;

            var max = float.MinValue;
            for (var classIndex = 0; classIndex < prediction.Length; classIndex++)
            {
                var probability = prediction[classIndex];
                if (probability > max)
                {
                    max = probability;
                    predictedClass = classIndex;
                }
            }

            return predictedClass;
        }

        static double PredictSingleClass(float[] prediction)
        {
            var predictedClass = 0;
            var singlePrediction = prediction.Single();

            if (singlePrediction > 0.5)
            {
                predictedClass = 1;
            }

            return predictedClass;
        }
    }
}
