using System.Collections.Generic;
using CNTK;

namespace SharpLearning.Cntk.Test
{
    public interface IMinibatchSource
    {
        UnorderedMapStreamInformationMinibatchData GetNextMinibatch(uint minibatchSizeInSamples, DeviceDescriptor device);
    }
}
