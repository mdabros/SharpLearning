using System.Runtime.InteropServices;

namespace SharpLearning.Containers.Views
{
    /// <summary>
    /// F64Matrix column view using pointers
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct F64MatrixColumnView
    {
        const int SizeOfType = sizeof(double);
        readonly double* m_dataPtr;
        readonly int m_rows;
        readonly int m_strideInBytes;

        /// <summary>
        /// Creates a column view from the provided data ptr, number of rows and stride
        /// </summary>
        /// <param name="dataPtr"></param>
        /// <param name="rows"></param>
        /// <param name="strideInBytes"></param>
        public F64MatrixColumnView(double* dataPtr, int rows, int strideInBytes)
        {
            m_dataPtr = dataPtr;
            m_rows = rows;
            m_strideInBytes = strideInBytes;
        }

        /// <summary>
        /// Gets the row item from the specified position
        /// </summary>
        /// <param name="row"></param>
        /// <returns></returns>
        public double this[int row]
        {
            get { return *RowPtr(row); }
            set { *RowPtr(row) = value; }
        }
        
        /// <summary>
        /// Gets the number of rows
        /// </summary>
   		public int Rows { get { return m_rows; } }

        /// <summary>
        /// Gets the interval of the column view
        /// </summary>
        public Interval1D Interval { get { return new Interval1D(0, m_rows); } }

        double* RowPtr(int row)
        {
            return (double*)((byte*)m_dataPtr + row * m_strideInBytes);
        }
    }
}
