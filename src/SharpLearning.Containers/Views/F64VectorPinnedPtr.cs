using System;
using System.Runtime.InteropServices;

namespace SharpLearning.Containers.Views
{
    /// <summary>
    /// Pinned pointer to F64Vector. Proper disposal required. Preferably use this in a Using statement
    /// 
    /// Using(var pinned = vector.GetPinnedPointer())
    /// {
    ///     var view = pinned.View();
    /// }
    /// 
    /// </summary>
    public unsafe struct F64VectorPinnedPtr : IDisposable
    {
        readonly int m_length;
        readonly GCHandle m_handle;
        double* m_ptr;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="v"></param>
        public F64VectorPinnedPtr(double[] v)
        {
            m_length = v.Length;
            m_handle = GCHandle.Alloc(v, GCHandleType.Pinned);
            m_ptr = (double*)m_handle.AddrOfPinnedObject().ToPointer();
        }

        /// <summary>
        /// Creates a view over the pinned F64Vector
        /// </summary>
        /// <returns></returns>
        public F64VectorView View()
        { 
            return new F64VectorView(m_ptr, m_length); 
        }

        /// <summary>
        /// 
        /// </summary>
        public void Dispose()
        {
            if (m_ptr != null)
            {
                m_ptr = null;
                m_handle.Free();
            }
        }
    }
}
