using MathNet.Numerics.LinearAlgebra.Storage;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Serialization;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

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
            m_neuralNet = model;
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
        /// Returns the raw variable importance. 
        /// Variable importance is calculated using the connection weights method
        /// also known as the Olden method. 
        /// </summary>
        /// <returns></returns>
        public double[] GetRawVariableImportance()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            throw new NotImplementedException();
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
