
namespace SharpLearning.Containers.Views
{
    /// <summary>
    /// View over an F64Matrix
    /// </summary>
    public unsafe struct F64MatrixView
    {
        const int SizeOfType = sizeof(double);
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
            RowCount = rows;
            ColumnCount = cols;
            m_strideInBytes = strideInBytes;
        }

        /// <summary>
        /// Gets the number of columns
        /// </summary>
        /// <value></value>
        public int ColumnCount { get; private set; }

        /// <summary>
        /// Gets the number of rows
        /// </summary>
        /// <value></value>
        public int RowCount { get; private set; }

        /// <summary>
        /// Gets the item at the specified position
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        public double At(int row, int col)
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
            get { return (double*)((byte*)m_dataPtr + (long)row * m_strideInBytes); }
        }

        /// <summary>
        /// Gets a column view of the specified column
        /// </summary>
        /// <param name="col"></param>
        /// <returns></returns>
        public F64MatrixColumnView ColumnView(int col)
        {
            return new F64MatrixColumnView(m_dataPtr + col, RowCount, m_strideInBytes);
        }

        /// <summary>
        /// Gets a sub-view over the specified interval
        /// </summary>
        /// <param name="subView"></param>
        /// <returns></returns>
        public F64MatrixView View(Interval2D subView)
        {
            return new F64MatrixView(GetSubViewDataPointer(subView), subView.Rows.Length, subView.Cols.Length, m_strideInBytes);
        }

        double* GetSubViewDataPointer(Interval2D subView)
        {
            return this[subView.Cols.FromInclusive] + subView.Rows.FromInclusive;
        }
    }
}
