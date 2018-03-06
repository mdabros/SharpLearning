using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;

namespace SharpLearning.Neural.Cntk
{
    public class CntkNeuralNetModel
    {
        readonly Function m_network;
        readonly DeviceDescriptor m_device;

        public CntkNeuralNetModel(Function network, DeviceDescriptor device)
        {
            if (network == null) { throw new ArgumentNullException("network"); }
            if (device == null) { throw new ArgumentNullException("device"); }
            m_network = network;
            m_device = device;
        }

        public IList<float> Predict(float[] observation)
        {
            var inputDimensions = m_network.Arguments[0].Shape;

            using (var batch = Value.CreateBatch<float>(inputDimensions, observation, m_device))
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
