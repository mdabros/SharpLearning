using System;

namespace SharpLearning.Backend
{
    public class TensorFlowBackend : IBackend
    {
        private object defaultDevice;

        public TensorFlowBackend(object defaultDevice)
        {
            this.defaultDevice = defaultDevice;
        }

        public void Dispose()
        {
            throw new NotImplementedException();
        }
    }
}