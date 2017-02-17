using System.Runtime.InteropServices;

namespace SharpLearning.Containers.Views
{
    /// <summary>
    /// View over F64Vector
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct F64VectorView
    {
        const int SizeOfType = sizeof(double);
        readonly double* m_ptr;
        readonly int m_length;

        /// <summary>
        /// Creates a view over an F64Vector
        /// </summary>
        /// <param name="dataPtr"></param>
        /// <param name="length"></param>
        public F64VectorView(double* dataPtr, int length)
        {
            m_ptr = dataPtr;
            m_length = length;
        }

        /// <summary>
        /// Gets the item from the specified position
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public double this[int index]
        {
            get { return m_ptr[index]; }
            set { m_ptr[index] = value; }
        }
        
        /// <summary>
        /// 
        /// </summary>
        public int Length { get { return m_length; } }

        /// <summary>
        /// Gets a subview over the specified interval
        /// </summary>
        /// <param name="interval"></param>
        /// <returns></returns>
        public F64VectorView View(Interval1D interval)
        {
            return new F64VectorView(GetSubViewDataPointer(interval), interval.Length);
        }
        
        double* GetSubViewDataPointer(Interval1D interval)
        {
            return m_ptr + interval.FromInclusive;
        }

        /// <summary>
        /// Gets the interval of the F64View
        /// </summary>
        public Interval1D Interval { get { return Interval1D.Create(0, m_length); } }
    }
}
