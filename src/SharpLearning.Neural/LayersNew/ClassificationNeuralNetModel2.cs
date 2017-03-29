using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    [Serializable]
    public sealed class ClassificationNeuralNetModel2 : IPredictorModel<double>, IPredictorModel<ProbabilityPrediction>
    {
        readonly NeuralNet2 m_neuralNet;
        readonly double[] m_targetNames;
        readonly TensorShape m_inputShape;
        readonly TensorShape m_outputShape;

        /// <summary>
        /// Classification neural net model
        /// </summary>
        /// <param name="model"></param>
        /// <param name="targetNames"></param>
        public ClassificationNeuralNetModel2(NeuralNet2 model, double[] targetNames)
        {
            if (model == null) { throw new ArgumentNullException("model"); }
            if (targetNames == null) { throw new ArgumentNullException("targetNames"); }

            m_neuralNet = model;
            m_targetNames = targetNames;

            var inputDimensions = new List<int> { 1 };
            inputDimensions.AddRange(m_neuralNet.Input.Dimensions.Skip(1));
            m_inputShape = new TensorShape(inputDimensions.ToArray());

            var outputDimensions = new List<int> { 1 };
            outputDimensions.AddRange(m_neuralNet.Output.Dimensions.Skip(1));
            m_outputShape = new TensorShape(outputDimensions.ToArray());

        }

        /// <summary>
        /// Predicts a single observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var tensorObservation = Tensor<double>.Build(observation, 
                m_inputShape.Dimensions.ToArray());

            var probabilities = m_neuralNet.Predict(tensorObservation).Data;

            var probability = 0.0;
            var prediction = 0.0;

            for (int i = 0; i < m_targetNames.Length; i++)
            {
                var currentProp = probabilities[i];
                if (currentProp > probability)
                {
                    prediction = m_targetNames[i];
                    probability = currentProp;
                }
            }
            return prediction;
        }

        /// <summary>
        /// Predicts a single observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        ProbabilityPrediction IPredictor<ProbabilityPrediction>.Predict(double[] observation)
        {
            return PredictProbability(observation);
        }


        /// <summary>
        /// Predicts a set of observations
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var rows = observations.RowCount;
            var cols = observations.ColumnCount;
            var predictions = new double[rows];
            var observation = new double[cols];
            for (int i = 0; i < rows; i++)
            {
                observations.Row(i, observation);
                predictions[i] = Predict(observation);
            }

            return predictions;
        }

        /// <summary>
        /// Predicts a single observation using the ensembled probabilities
        /// Note this can yield a different result than using regular predict
        /// Usally this will be a more accurate predictions
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            var tensorObservation = Tensor<double>.Build(observation,
                m_inputShape.Dimensions.ToArray());

            var probabilities = m_neuralNet.Predict(tensorObservation).Data;
            var probabilityDictionary = new Dictionary<double, double>();

            var probability = 0.0;
            var prediction = 0.0;

            for (int i = 0; i < m_targetNames.Length; i++)
            {

                probabilityDictionary.Add(m_targetNames[i], probabilities[i]);
                if (probabilities[i] > probability)
                {
                    probability = probabilities[i];
                    prediction = m_targetNames[i];
                }
            }

            return new ProbabilityPrediction(prediction, probabilityDictionary);
        }

        /// <summary>
        /// Predicts a set of obervations using the ensembled probabilities
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public ProbabilityPrediction[] PredictProbability(F64Matrix observations)
        {
            var rows = observations.RowCount;
            var cols = observations.ColumnCount;
            var predictions = new ProbabilityPrediction[rows];
            var observation = new double[cols];
            for (int i = 0; i < rows; i++)
            {
                observations.Row(i, observation);
                predictions[i] = PredictProbability(observation);
            }

            return predictions;
        }

        /// <summary>
        /// Variable importance is currently not supported by Neural Net.
        /// Returns 0.0 for all features.
        /// </summary>
        /// <returns></returns>
        public double[] GetRawVariableImportance()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Variable importance is currently not supported by Neural Net.
        /// Returns 0.0 for all features.
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            throw new NotImplementedException();
        }
    }
}
