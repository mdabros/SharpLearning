using System;
using System.Linq;
using SharpLearning.Containers.Views;
using SharpLearning.Containers.Extensions;

namespace SharpLearning.Containers.Matrices
{
    /// <summary>
    /// Matrix of doubles
    /// </summary>
    /// <remarks>Can be implicitly converted from double[][]</remarks>
    public sealed unsafe class F64Matrix : IMatrix<double>, IEquatable<F64Matrix>
    {
        double[] m_featureArray;

        /// <summary>
        /// Creates a zero-matrix with the specified number of rows and cols
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="cols"></param>
        public F64Matrix(int rows, int cols)
            : this(new double[rows * cols], rows, cols)
        {
        }

        /// <summary>
        /// Creates a matrix from the provided values with the specified rows and cols 
        /// </summary>
        /// <param name="values"></param>
        /// <param name="rows"></param>
        /// <param name="cols"></param>
        public F64Matrix(double[] values, int rows, int cols)
        {
            if (values == null) { throw new ArgumentNullException("values"); }
            if (values.Length != rows * cols) { throw new ArgumentException("feature array length does not match row * cols"); }
            if (rows < 1) { throw new ArgumentException("matrix must have at least 1 row"); }
            if (cols < 1) { throw new ArgumentException("matrix must have at least 1 col"); }
            
            m_featureArray = values;
            RowCount = rows;
            ColumnCount = cols;
        }

        /// <summary>
        /// Gets the item at the specified position
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        public double At(int row, int col)
        {
            var rowOffSet = row * ColumnCount;
            var item = m_featureArray[rowOffSet + col];

            return item;
        }

        /// <summary>
        /// Sets the item at the specified posistion
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <param name="item"></param>
        public void At(int row, int col, double item)
        {
            var rowOffSet = row * ColumnCount;
            m_featureArray[rowOffSet + col] = item;
        }

        /// <summary>
        /// Access the matrix like a 2D array
        /// </summary>
        /// <param name="col"></param>
        /// <param name="row"></param>
        /// <returns></returns>
        public double this[int row, int col]
        {
            get { return m_featureArray[row * ColumnCount + col]; }
            set { m_featureArray[row * ColumnCount + col] = value; }
        }

        /// <summary>
        /// Gets the specified row
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public double[] Row(int index)
        {
            var row = new double[ColumnCount];
            Row(index, row);
            return row;
        }

        /// <summary>
        /// gets the specified row. 
        /// The values are copied to the provided row array.
        /// </summary>
        /// <param name="index"></param>
        /// <param name="row"></param>
        public void Row(int index, double[] row)
        {
            var rowOffSet = index * ColumnCount;

            for (int i = 0; i < ColumnCount; i++)
            {
                row[i] = m_featureArray[rowOffSet + i];
            }
        }

        /// <summary>
        /// Gets the specified column
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public double[] Column(int index)
        {
            var col = new double[RowCount];

            for (int i = 0; i < RowCount; i++)
            {
                col[i] = m_featureArray[ColumnCount * i + index];
            }

            return col;
        }

        /// <summary>
        /// Gets the specified column.
        /// The values are copied to the provided column array.
        /// </summary>
        /// <param name="index"></param>
        /// <param name="col"></param>
        public void Column(int index, double[] col)
        {
            for (int i = 0; i < RowCount; i++)
            {
                col[i] = m_featureArray[ColumnCount * i + index];
            }
        }

        /// <summary>
        /// Gets the specified rows as a matrix
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        public IMatrix<double> Rows(params int[] indices)
        {
            var rowCount = indices.Length;
            var subFeatureArray = new double[rowCount * ColumnCount];

            for (int i = 0; i < indices.Length; i++)
            {
                var rowOffSet = ColumnCount * indices[i];
                var subRowOffSet = ColumnCount * i;
                for (int j = 0; j < ColumnCount; j++)
                {
                    subFeatureArray[subRowOffSet + j] = m_featureArray[rowOffSet + j];
                }
            }

            return new F64Matrix(subFeatureArray, indices.Length, ColumnCount);
        }

        /// <summary>
        /// Gets the specified rows as a matrix. 
        /// Output is copied to the provided matrix
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="output"></param>
        /// <returns></returns>
        public void Rows(int[] indices, IMatrix<double> output)
        {
            var rowCount = indices.Length;
            var subFeatureArray = output.Data();

            for (int i = 0; i < indices.Length; i++)
            {
                var rowOffSet = ColumnCount * indices[i];
                var subRowOffSet = ColumnCount * i;
                for (int j = 0; j < ColumnCount; j++)
                {
                    subFeatureArray[subRowOffSet + j] = m_featureArray[rowOffSet + j];
                }
            }
        }

        /// <summary>
        /// Gets the specified columns as a matrix
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        public IMatrix<double> Columns(params int[] indices)
        {
            var subFeatureCount = indices.Length;
            var subFeatureArray = new double[RowCount * subFeatureCount];
            for (int i = 0; i < RowCount; i++)
            {
                var rowOffSet = ColumnCount * i;
                var subRowOffSet = subFeatureCount * i;
                for (int j = 0; j < indices.Length; j++)
                {
                    subFeatureArray[subRowOffSet + j] = m_featureArray[rowOffSet + indices[j]];
                }
            }

            return new F64Matrix(subFeatureArray, RowCount, indices.Length);
        }

        /// <summary>
        /// Gets the specified rows as a matrix. 
        /// Output is copied to the provided matrix
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="output"></param>
        /// <returns></returns>
        public void Columns(int[] indices, IMatrix<double> output)
        {
            var subFeatureCount = indices.Length;
            var subFeatureArray = output.Data();
            for (int i = 0; i < RowCount; i++)
            {
                var rowOffSet = ColumnCount * i;
                var subRowOffSet = subFeatureCount * i;
                for (int j = 0; j < indices.Length; j++)
                {
                    subFeatureArray[subRowOffSet + j] = m_featureArray[rowOffSet + indices[j]];
                }
            }
        }

        /// <summary>
        /// Gets the 1-d array containing all the values of the matrix
        /// </summary>
        /// <returns></returns>
        public double[] Data()
        {
            return m_featureArray;
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
        /// Gets a pinned pointer to the F64Matrix
        /// </summary>
        /// <returns></returns>
        public F64MatrixPinnedPtr GetPinnedPointer()
        {
            return new F64MatrixPinnedPtr(this);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(F64Matrix other)
        {
            if (this.RowCount != other.RowCount) { return false; }
            if (this.ColumnCount != other.ColumnCount) { return false; }
            if (!this.Data().SequenceEqual(other.Data())) { return false; }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (obj is F64Matrix other && this.Equals(other))
            {
                return true;
            }

            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            unchecked // Overflow is fine, just wrap
            {
                int hash = 17;
                hash = hash * 23 + m_featureArray.GetHashCode();
                hash = hash * 23 + ColumnCount.GetHashCode();
                hash = hash * 23 + RowCount.GetHashCode();

                return hash;
            }
        }

        public static implicit operator F64Matrix(double[][] b) => b.ToF64Matrix();
    }
}
