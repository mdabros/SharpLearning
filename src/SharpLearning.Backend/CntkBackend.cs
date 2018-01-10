using System;

namespace SharpLearning.Backend
{
    public class CntkBackend : IBackend
    {
        private object defaultDevice;

        public CntkBackend(object defaultDevice)
        {
            this.defaultDevice = defaultDevice;
        }

        public void Dispose()
        {
            throw new NotImplementedException();
        }
    }
}