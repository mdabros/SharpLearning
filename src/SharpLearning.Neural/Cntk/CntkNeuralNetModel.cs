using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Neural.Cntk
{
    public class CntkNeuralNetModel : IPredictorModel<double>, IPredictorModel<ProbabilityPrediction>
    {
        readonly Function m_network;
        readonly DeviceDescriptor m_device;
        readonly double[] m_targetNames;

        public CntkNeuralNetModel(Function network, DeviceDescriptor device,
            double[] targetNames)
        {
            if (network == null) { throw new ArgumentNullException("network"); }
            if (device== null) { throw new ArgumentNullException("device"); }
            if (targetNames == null) { throw new ArgumentNullException("targetNames"); }
            m_network = network;
            m_device = device;
            m_targetNames = targetNames;
        }

        public double Predict(double[] observation)
        {
            var probabilities = EvaluteProbabilities(observation);

            var probability = 0.0;
            var prediction = 0.0;

            for (int i = 0; i < m_targetNames.Length; i++)
            {
                var currentProp = (double)probabilities[i];
                if (currentProp > probability)
                {
                    prediction = m_targetNames[i];
                    probability = currentProp;
                }
            }

            return prediction;
        }

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

        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            var probabilities = EvaluteProbabilities(observation);
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

        ProbabilityPrediction IPredictor<ProbabilityPrediction>.Predict(double[] observation)
        {
            return PredictProbability(observation);
        }

        public double[] GetRawVariableImportance()
        {
            throw new NotImplementedException();
        }

        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            throw new NotImplementedException();
        }

        IList<float> EvaluteProbabilities(double[] observation)
        {
            var inputDimensions = m_network.Arguments[0].Shape;
            var features = new float[observation.Length];
            CntkUtils.ConvertArray(observation, features);

            using (var batch = Value.CreateBatch<float>(inputDimensions, features, m_device))
            {
                var inputDataMap = new Dictionary<Variable, Value>() { { m_network.Arguments[0], batch } };
                var outputVar = m_network.Output;

                var outputDataMap = new Dictionary<Variable, Value>() { { outputVar, null } };
                m_network.Evaluate(inputDataMap, outputDataMap, m_device);
                var outputVal = outputDataMap[outputVar];
                var predictions = outputVal.GetDenseData<float>(outputVar);
                return predictions.Single();
            }
        }
    }
}
