using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Containers.Matrices
{
    /// <summary>
    /// Matrix of strings
    /// </summary>
    public sealed class StringMatrix : IMatrix<string>, IEquatable<StringMatrix>
    {
        string[] m_featureArray;
        readonly int m_rows;
        readonly int m_cols;

        /// <summary>
        /// Creates a empty string matrix with the specified number of rows and cols
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="cols"></param>
        public StringMatrix(int rows, int cols)
            : this(new string[rows * cols], rows, cols)
        {
        }

        /// <summary>
        /// Creates a matrix from the provided values with the specified rows and cols 
        /// </summary>
        /// <param name="values"></param>
        /// <param name="rows"></param>
        /// <param name="cols"></param>
        public StringMatrix(string[] values, int rows, int cols)
        {
            if (values == null) { throw new ArgumentNullException("values"); }
            if (values.Length != rows * cols) { throw new ArgumentException("feature array length does not match row * cols"); }
            if (rows < 1) { throw new ArgumentException("matrix must have at least 1 row"); }
            if (cols < 1) { throw new ArgumentException("matrix must have at least 1 col"); }

            m_featureArray = values;
            m_rows = rows;
            m_cols = cols;
        }

        /// <summary>
        /// Gets the item at the specified position
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        public string GetItemAt(int row, int col)
        {
            var rowOffSet = row * m_cols;
            var item = m_featureArray[rowOffSet + col];

            return item;
        }

        /// <summary>
        /// Sets the item at the specified posistion
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <param name="item"></param>
        public void SetItemAt(int row, int col, string item)
        {
            var rowOffSet = row * m_cols;
            m_featureArray[rowOffSet + col] = item;
        }

        /// <summary>
        /// Access the matrix like a 2D array
        /// </summary>
        /// <param name="col"></param>
        /// <param name="row"></param>
        /// <returns></returns>
        public string this[int col, int row]
        {
            get { return m_featureArray[row * m_cols + col]; }
            set { m_featureArray[row * m_cols + col] = value; }
        }

        /// <summary>
        /// Gets the specified row
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public string[] GetRow(int index)
        {
            var row = new string[m_cols];
            var rowOffSet = index * m_cols;

            for (int i = 0; i < m_cols; i++)
            {
                row[i] = m_featureArray[rowOffSet + i];
            }

            return row;
        }

        /// <summary>
        /// gets the specified row. 
        /// The values are copied to the provided row array.
        /// </summary>
        /// <param name="index"></param>
        /// <param name="row"></param>
        public void GetRow(int index, string[] row)
        {
            var rowOffSet = index * m_cols;

            for (int i = 0; i < m_cols; i++)
            {
                row[i] = m_featureArray[rowOffSet + i];
            }
        }

        /// <summary>
        /// Gets the specified column
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public string[] GetColumn(int index)
        {
            var col = new string[m_rows];

            for (int i = 0; i < m_rows; i++)
            {
                col[i] = m_featureArray[m_cols * i + index];
            }

            return col;
        }

        /// <summary>
        /// Gets the specified column.
        /// The values are copied to the provided column array.
        /// </summary>
        /// <param name="index"></param>
        /// <param name="col"></param>
        public void GetColumn(int index, string[] col)
        {
            for (int i = 0; i < m_rows; i++)
            {
                col[i] = m_featureArray[m_cols * i + index];
            }
        }

        /// <summary>
        /// Gets the specified rows as a matrix
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        public IMatrix<string> GetRows(params int[] indices)
        {
            var rowCount = indices.Length;
            var subFeatureArray = new string[rowCount * m_cols];

            for (int i = 0; i < indices.Length; i++)
            {
                var rowOffSet = m_cols * indices[i];
                var subRowOffSet = m_cols * i;
                for (int j = 0; j < m_cols; j++)
                {
                    subFeatureArray[subRowOffSet + j] = m_featureArray[rowOffSet + j];
                }
            }

            return new StringMatrix(subFeatureArray, indices.Length, m_cols);
        }

        /// <summary>
        /// Gets the specified columns as a matrix
        /// </summary>
        /// <param name="indices"></param>
        /// <returns></returns>
        public IMatrix<string> GetColumns(params int[] indices)
        {
            var subFeatureCount = indices.Length;
            var subFeatureArray = new string[m_rows * subFeatureCount];
            for (int i = 0; i < m_rows; i++)
            {
                var rowOffSet = m_cols * i;
                var subRowOffSet = subFeatureCount * i;
                for (int j = 0; j < indices.Length; j++)
                {
                    subFeatureArray[subRowOffSet + j] = m_featureArray[rowOffSet + indices[j]];
                }
            }

            return new StringMatrix(subFeatureArray, m_rows, indices.Length);
        }

        /// <summary>
        /// Gets the 1-d array containing all the values of the matrix
        /// </summary>
        /// <returns></returns>
        public string[] GetFeatureArray()
        {
            return m_featureArray;
        }

        /// <summary>
        /// Gets the number of columns
        /// </summary>
        /// <returns></returns>
        public int GetNumberOfColumns()
        {
            return m_cols;
        }

        /// <summary>
        /// Gets the number of rows
        /// </summary>
        /// <returns></returns>
        public int GetNumberOfRows()
        {
            return m_rows;
        }

        public bool Equals(StringMatrix other)
        {
            if (this.GetNumberOfRows() != other.GetNumberOfRows()) { return false; }
            if (this.GetNumberOfColumns() != other.GetNumberOfColumns()) { return false; }
            if (!this.GetFeatureArray().SequenceEqual(other.GetFeatureArray())) { return false; }

            return true;
        }

        public override bool Equals(object obj)
        {
            StringMatrix other = obj as StringMatrix;
            if (other != null && Equals(other))
            {
                return true;
            }

            return false;
        }

        public override int GetHashCode()
        {
            unchecked // Overflow is fine, just wrap
            {
                int hash = 17;
                hash = hash * 23 + m_featureArray.GetHashCode();
                hash = hash * 23 + m_cols.GetHashCode();
                hash = hash * 23 + m_rows.GetHashCode();

                return hash;
            }
        }
    }
}
