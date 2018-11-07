using System;
using CNTK;

namespace CntkCatalyst
{
    public class CntkMinibatchSource : IMinibatchSource
    {
        readonly MinibatchSource m_minibatchSource;

        public CntkMinibatchSource(MinibatchSource minibatchSource, string featuresName, string labelsName)
        {
            m_minibatchSource = minibatchSource ?? throw new ArgumentNullException(nameof(minibatchSource));
            FeaturesName = featuresName;
            TargetsName = labelsName;
        }

        public string FeaturesName { get; }
        public string TargetsName { get; }

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
