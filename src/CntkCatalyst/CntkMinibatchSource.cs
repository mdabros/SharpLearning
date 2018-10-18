using System;
using CNTK;

namespace CntkCatalyst
{
    public class CntkMinibatchSource : IMinibatchSource
    {
        readonly MinibatchSource m_minibatchSource;
        readonly string m_featuresName;
        readonly string m_labelsName;

        public CntkMinibatchSource(MinibatchSource minibatchSource, string featuresName, string labelsName)
        {
            m_minibatchSource = minibatchSource ?? throw new ArgumentNullException(nameof(minibatchSource));
            m_featuresName = featuresName;
            m_labelsName = labelsName;
        }

        public string FeaturesName => m_featuresName;
        public string LabelsName => m_labelsName;

        public UnorderedMapStreamInformationMinibatchData GetNextMinibatch(uint minibatchSizeInSamples, 
            DeviceDescriptor device)
        {
            return m_minibatchSource.GetNextMinibatch(minibatchSizeInSamples, device);            
        }

        public StreamInformation StreamInfo(string streamName)
        {
            return m_minibatchSource.StreamInfo(streamName);
        }
    }
}
