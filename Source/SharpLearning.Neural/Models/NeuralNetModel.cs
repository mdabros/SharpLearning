using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Neural.Models
{
    /// <summary>
    /// Base neuralnet model
    /// </summary>
    [Serializable]
    public class NeuralNetModel
    {
        readonly List<Matrix<float>> m_weights;
        readonly List<Vector<float>> m_intercepts;

        readonly IActivation m_hiddenActivation;
        readonly IActivation m_outputActivation;

        readonly int m_layes;

        /// <summary>
        /// Iterations used for training the model
        /// </summary>
        public readonly int Iterations;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="weights"></param>
        /// <param name="intercepts"></param>
        /// <param name="hiddenActivation"></param>
        /// <param name="outputActivation"></param>
        /// <param name="iterations"></param>
        public NeuralNetModel(List<Matrix<float>> weights, List<Vector<float>> intercepts, IActivation hiddenActivation, IActivation outputActivation, int iterations)
        {
            m_weights = weights;
            m_intercepts = intercepts;
            m_hiddenActivation = hiddenActivation;
            m_outputActivation = outputActivation;
            m_layes = weights.Count + 1;
            Iterations = iterations;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="withOutputActivation"></param>
        /// <returns></returns>
        public Matrix<float> ForwardPass(Matrix<float> observations, bool withOutputActivation = true)
        {
            var output = observations;
            for (int i = 0; i < m_layes - 1; i++)
            {
                output = output.Multiply(m_weights[i]);
                output.AddRowWise(m_intercepts[i], output);

                if ((i + 1) != m_layes - 1)
                {
                    m_hiddenActivation.Activation(output);
                }
            }

            if (withOutputActivation)
            {
                m_outputActivation.Activation(output);
            }

            return output;
        }

        /// <summary>
        /// Returns the raw variable importance. 
        /// Variable importance is calculated using the connection weights method
        /// also known as the Olden method. 
        /// </summary>
        /// <returns></returns>
        public double[] GetRawVariableImportance()
        {
            Matrix<float> result = m_weights[0].Multiply(m_weights[1]);

            if(m_weights.Count > 2)
            {
                for (int i = 2; i < m_weights.Count; i++)
                {
                    result = result.Multiply(m_weights[i]);
                }
            }

            return result.RowAbsoluteSums().Select(d => (double)d).ToArray();
        }

        /// <summary>
        /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            var rawVariableImportance = GetRawVariableImportance();
            var max = rawVariableImportance.Max();

            var scaledVariableImportance = rawVariableImportance
                .Select(v => (v / max) * 100.0)
                .ToArray();

            return featureNameToIndex.ToDictionary(kvp => kvp.Key, kvp => scaledVariableImportance[kvp.Value])
                        .OrderByDescending(kvp => kvp.Value)
                        .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }
    }
}
