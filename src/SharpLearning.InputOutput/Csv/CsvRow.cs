using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.InputOutput.Csv
{
    /// <summary>
    /// CsvRow holding the row values and column name to index
    /// </summary>
    public class CsvRow
    {
        /// <summary>
        /// Values
        /// </summary>
        public readonly string[] Values;
        
        /// <summary>
        /// Column name to index
        /// </summary>
        public readonly Dictionary<string, int> ColumnNameToIndex;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="columnNameToIndex"></param>
        /// <param name="data"></param>
        public CsvRow(Dictionary<string, int> columnNameToIndex, params string[] data)
        {
            if (data == null) { throw new ArgumentException("row"); }
            if (columnNameToIndex == null) { throw new ArgumentException("columnNameToIndex"); }
            if (data.Length != columnNameToIndex.Count) { throw new ArgumentException("data and columNameToIndex lengths does not match"); }
            Values = data;
            ColumnNameToIndex = columnNameToIndex;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(CsvRow other)
        {
            if (!this.Values.SequenceEqual(other.Values))
                return false;

            if (!this.ColumnNameToIndex.SequenceEqual(other.ColumnNameToIndex))
                return false;

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (obj is CsvRow other)
            {
                return Equals(other);
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
                // Suitable nullity checks etc, of course :)
                hash = hash * 23 + Values.GetHashCode();
                hash = hash * 23 + ColumnNameToIndex.GetHashCode();

                return hash;
            }
        }
    }
}
