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
        public readonly string[] Values;
        public readonly Dictionary<string, int> ColumnNameToIndex;

        public CsvRow(string[] data, Dictionary<string, int> columnNameToIndex)
        {
            if (data == null) { throw new ArgumentException("row"); }
            if (columnNameToIndex == null) { throw new ArgumentException("columnNameToIndex"); }
            if (data.Length != columnNameToIndex.Count) { throw new ArgumentException("data and columNameToIndex lengths does not match"); }
            Values = data;
            ColumnNameToIndex = columnNameToIndex;
        }

        public bool Equals(CsvRow other)
        {
            if (!this.Values.SequenceEqual(other.Values))
                return false;

            if (!this.ColumnNameToIndex.SequenceEqual(other.ColumnNameToIndex))
                return false;

            return true;
        }

        public override bool Equals(object obj)
        {
            CsvRow other = obj as CsvRow;
            if (other != null)
            {
                return Equals(other);
            }

            return false;
        }

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
