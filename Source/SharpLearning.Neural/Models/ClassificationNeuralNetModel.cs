using MathNet.Numerics.LinearAlgebra.Storage;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Serialization;
using System;
using System.Collections.Generic;
using System.IO;

namespace SharpLearning.Neural.Models
{

    /// <summary>
    /// Classification neural net model.
    /// </summary>
    [Serializable]
    public sealed class ClassificationNeuralNetModel : IPredictorModel<double>, IPredictorModel<ProbabilityPrediction>
    {
        readonly NeuralNet m_neuralNet;
        readonly double[] m_targetNames;

        /// <summary>
        /// Classification neural net model
        /// </summary>
        /// <param name="model"></param>
        /// <param name="targetNames"></param>
        public ClassificationNeuralNetModel(NeuralNet model, double[] targetNames)
        {
            if (model == null) { throw new ArgumentNullException("model"); }
            if (targetNames == null) { throw new ArgumentNullException("targetNames"); }

            m_neuralNet = model;
            m_targetNames = targetNames;
        }

        /// <summary>
        /// Predicts a single observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var mObservation = observation
                .ConvertDoubleArray();

            var probabilities = m_neuralNet.Forward(mObservation);

            var probability = 0.0;
            var prediction = 0.0;

            for (int i = 0; i < m_targetNames.Length; i++)
            {
                var currentProp = (double)probabilities[0, i];
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
            var rows = observations.RowCount();
            var cols = observations.ColumnCount();
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
            var mObservation = observation
                .ConvertDoubleArray();

            var probabilities = m_neuralNet.Forward(mObservation);
            var probabilityDictionary = new Dictionary<double, double>();

            var probability = 0.0;
            var prediction = 0.0;

            for (int i = 0; i < m_targetNames.Length; i++)
            {

                probabilityDictionary.Add(m_targetNames[i], probabilities[0, i]);
                if (probabilities[0, i] > probability)
                {
                    probability = probabilities[0, i];
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
            var rows = observations.RowCount();
            var cols = observations.ColumnCount();
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
            return m_neuralNet.GetRawVariableImportance();
        }

        /// <summary>
        /// Variable importance is currently not supported by Neural Net.
        /// Returns 0.0 for all features.
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            return m_neuralNet.GetVariableImportance(featureNameToIndex);
        }

        /// <summary>
        /// Outputs a string representation of the neural net.
        /// Neural net must be initialized before the dimensions are correct.
        /// </summary>
        /// <returns></returns>
        public string GetLayerDimensions()
        {
            return m_neuralNet.GetLayerDimensions();
        }

        /// <summary>
        /// Loads a ClassificationNeuralNetModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static ClassificationNeuralNetModel Load(Func<TextReader> reader)
        {
            var types = new Type[]
            {
                typeof(DenseVectorStorage<float>),
                typeof(DenseColumnMajorMatrixStorage<float>)
            };

            return new GenericXmlDataContractSerializer(types)
                .Deserialize<ClassificationNeuralNetModel>(reader);
        }

        /// <summary>
        /// Saves the ClassificationNeuralNetModel.
        /// </summary>
        /// <param name="writer"></param>
        public void Save(Func<TextWriter> writer)
        {
            var types = new Type[]
            {
                typeof(DenseVectorStorage<float>),
                typeof(DenseColumnMajorMatrixStorage<float>)
            };

            new GenericXmlDataContractSerializer(types)
                .Serialize(this, writer);
        }
    }
}
