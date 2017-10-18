using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;
using SharpLearning.Common.Interfaces;

namespace SharpLearning.Neural.Cntk
{
    public class CntkNeuralNetModel : IPredictorModel<double>
    {
        readonly Function m_network;
        readonly DeviceDescriptor m_device;

        public CntkNeuralNetModel(Function network, DeviceDescriptor device)
        {
            if (network == null) { throw new ArgumentNullException("network"); }
            if (device== null) { throw new ArgumentNullException("device"); }
            m_network = network;
            m_device = device;
        }

        public double Predict(double[] observation)
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
                var actual = outputVal.GetDenseData<float>(outputVar);

                // needs modification
                return (double)actual.Single().First();
            }
        }

        public double[] GetRawVariableImportance()
        {
            throw new NotImplementedException();
        }

        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            throw new NotImplementedException();
        }
    }
}
