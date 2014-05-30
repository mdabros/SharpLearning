
namespace SharpLearning.Containers.Views
{
    /// <summary>
    /// Creates a view over an F64Matrix
    /// </summary>
    public unsafe struct F64MatrixView
    {
        const int SizeOfType = sizeof(double);
        readonly int m_rows;
        readonly int m_cols;
        readonly int m_strideInBytes;
        readonly double* m_dataPtr;
        
        /// <summary>
        /// Creates a view over an F64Matrix
        /// </summary>
        /// <param name="dataPtr"></param>
        /// <param name="rows"></param>
        /// <param name="cols"></param>
        public F64MatrixView(double* dataPtr, int rows, int cols)
            : this(dataPtr, rows, cols, SizeOfType * cols)
        {
        }

        /// <summary>
        /// Creates a view over an F64Matrix
        /// </summary>
        /// <param name="dataPtr"></param>
        /// <param name="rows"></param>
        /// <param name="cols"></param>
        /// <param name="strideInBytes"></param>
        public F64MatrixView(double* dataPtr, int rows, int cols, int strideInBytes)
        {
            m_dataPtr = dataPtr;
            m_rows = rows;
            m_cols = cols;
            m_strideInBytes = strideInBytes;
        }

        double* GetSubViewDataPointer(Interval2D subView)
        {
            return this[subView.Cols.FromInclusive] + subView.Rows.FromInclusive;
        }

        /// <summary>
        /// Gets the item at the specified position
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        public double GetItemAt(int row, int col)
        {
            return this[row][col];
        }

        /// <summary>
        /// Gets a pointer to the specified position
        /// </summary>
        /// <param name="row"></param>
        /// <returns></returns>
        public double* this[int row]
        {
            get { return (double*)((byte*)m_dataPtr + row * m_strideInBytes); }
        }

        /// <summary>
        /// Gets a column view of the specified column
        /// </summary>
        /// <param name="col"></param>
        /// <returns></returns>
        public F64MatrixColumnView ColumnView(int col)
        {
            return new F64MatrixColumnView(m_dataPtr + col, m_rows, m_strideInBytes);
        }

        /// <summary>
        /// Gets a subview over the specified interval
        /// </summary>
        /// <param name="subView"></param>
        /// <returns></returns>
        public F64MatrixView View(Interval2D subView)
        {
            return new F64MatrixView(GetSubViewDataPointer(subView), subView.Rows.Length, subView.Cols.Length, m_strideInBytes);
        }
    }
}
