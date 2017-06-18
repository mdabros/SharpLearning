using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    [Serializable]
    public sealed class RegressionNeuralNetModel2 : IPredictorModel<double>
    {
        readonly NeuralNet2 m_neuralNet;

        readonly TensorShape m_inputShape;
        readonly TensorShape m_outputShape;

        /// <summary>
        /// Regression neural net model
        /// </summary>
        /// <param name="model"></param>
        public RegressionNeuralNetModel2(NeuralNet2 model)
        {
            if (model == null) { throw new ArgumentNullException("model"); }

            m_neuralNet = model;

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

            var prediction = m_neuralNet.Predict(tensorObservation).Data.Single();

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
