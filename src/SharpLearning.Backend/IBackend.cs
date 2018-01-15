using System;

namespace SharpLearning.Backend
{
    public interface IBackend : IDisposable
    {
        DeviceType DefaultDeviceType { get; }
    }
}