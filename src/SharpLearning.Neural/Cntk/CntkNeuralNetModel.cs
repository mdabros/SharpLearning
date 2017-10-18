using System;
using System.Collections.Generic;
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
            throw new NotImplementedException();
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
