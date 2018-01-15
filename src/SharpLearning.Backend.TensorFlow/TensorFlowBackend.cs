using System;
//using TensorFlow; // Does not work

namespace SharpLearning.Backend
{
    public class TensorFlowBackend : IBackend
    {
        //TFGraph m_graph;

        public TensorFlowBackend(DeviceType defaultDeviceType)
        {
            DefaultDeviceType = defaultDeviceType;
            //m_graph = new TFGraph();
        }

        public DeviceType DefaultDeviceType { get; }

        private void DisposeManagedResources()
        {
            // TODO: Insert managed dispose here
        }

        #region Dispose
        public void Dispose()
        {
            Dispose(true);
        }

        // Dispose(bool disposing) executes in two distinct scenarios.
        // If disposing equals true, the method has been called directly
        // or indirectly by a user's code. Managed and unmanaged resources
        // can be disposed.
        // If disposing equals false, the method has been called by the 
        // runtime from inside the finalizer and you should not reference 
        // other objects. Only unmanaged resources can be disposed.
        protected void Dispose(bool disposing)
        {
            // Dispose only if we have not already disposed.
            if (!m_disposed)
            {
                // If disposing equals true, dispose all managed and unmanaged resources.
                // I.e. dispose managed resources only if true, unmanaged always.
                if (disposing)
                {
                    DisposeManagedResources();
                }

                // Call the appropriate methods to clean up unmanaged resources here.
                // If disposing is false, only the following code is executed.
            }
            m_disposed = true;
        }

        private volatile bool m_disposed = false;
        #endregion
    }
}