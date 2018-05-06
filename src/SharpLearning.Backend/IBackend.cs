using System;

namespace SharpLearning.Backend
{
    public interface IBackend : IDisposable
    {
        IGraph CreateGraph();
    }
}