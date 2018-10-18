using System;
using CNTK;

namespace CntkCatalyst
{
    public class CntkMinibatchSource : IMinibatchSource
    {
        readonly MinibatchSource m_minibatchSource;

        public CntkMinibatchSource(MinibatchSource minibatchSource)
        {
            m_minibatchSource = minibatchSource ?? throw new ArgumentNullException(nameof(minibatchSource));
        }

        public UnorderedMapStreamInformationMinibatchData GetNextMinibatch(uint minibatchSizeInSamples, 
            DeviceDescriptor device)
        {
            return m_minibatchSource.GetNextMinibatch(minibatchSizeInSamples, device);
        }
    }
}
