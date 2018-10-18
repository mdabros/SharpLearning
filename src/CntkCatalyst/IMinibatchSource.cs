using CNTK;

namespace CntkCatalyst
{
    public interface IMinibatchSource
    {
        UnorderedMapStreamInformationMinibatchData GetNextMinibatch(uint minibatchSizeInSamples, 
            DeviceDescriptor device);

        StreamInformation StreamInfo(string streamName);

        string FeaturesName { get; }
        string TargetsName { get; }
    }
}