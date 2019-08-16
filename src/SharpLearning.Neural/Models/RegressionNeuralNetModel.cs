using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Storage;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Serialization;

namespace SharpLearning.Neural.Models
{

    /// <summary>
    /// Regression neural net model.
    /// </summary>
    [Serializable]
    public sealed class RegressionNeuralNetModel : IPredictorModel<double>
    {
        readonly NeuralNet m_neuralNet;

        /// <summary>
        /// Regression neural net model
        /// </summary>
        /// <param name="model"></param>
        public RegressionNeuralNetModel(NeuralNet model)
        {
            m_neuralNet = model ?? throw new ArgumentNullException(nameof(model));
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

            var prediction = (double)m_neuralNet.Forward(mObservation)
                .ToColumnMajorArray().Single();

            return prediction;
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
        /// Variable importance is currently not supported by Neural Net.
        /// Returns 0.0 for all features.
        /// </summary>
        /// <returns></returns>
        public double[] GetRawVariableImportance() => m_neuralNet.GetRawVariableImportance();

        /// <summary>
        /// Variable importance is currently not supported by Neural Net.
        /// Returns 0.0 for all features.
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex) 
            => m_neuralNet.GetVariableImportance(featureNameToIndex);

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
        /// Loads a RegressionNeuralNetModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static RegressionNeuralNetModel Load(Func<TextReader> reader)
        {
            var types = new Type[]
            {
                typeof(DenseVectorStorage<float>),
                typeof(DenseColumnMajorMatrixStorage<float>)
            };

            return new GenericXmlDataContractSerializer(types)
                .Deserialize<RegressionNeuralNetModel>(reader);
        }

        /// <summary>
        /// Saves the RegressionNeuralNetModel.
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
