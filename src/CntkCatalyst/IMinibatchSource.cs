using CNTK;

namespace CntkCatalyst
{
    public interface IMinibatchSource
    {
        UnorderedMapStreamInformationMinibatchData GetNextMinibatch(uint minibatchSizeInSamples, DeviceDescriptor device);
    }
}