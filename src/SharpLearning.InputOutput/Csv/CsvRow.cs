﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.InputOutput.Csv;

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

    public CsvRow(Dictionary<string, int> columnNameToIndex, params string[] data)
    {
        if (data == null) { throw new ArgumentException("row"); }
        if (columnNameToIndex == null) { throw new ArgumentNullException(nameof(columnNameToIndex)); }
        if (data.Length != columnNameToIndex.Count) { throw new ArgumentException("data and columNameToIndex lengths does not match"); }
        Values = data;
        ColumnNameToIndex = columnNameToIndex;
    }

    public bool Equals(CsvRow other)
    {
        if (!Values.SequenceEqual(other.Values))
        {
            return false;
        }

        return ColumnNameToIndex.SequenceEqual(other.ColumnNameToIndex);
    }

    public override bool Equals(object obj)
    {
        return obj is CsvRow other && Equals(other);
    }

    public override int GetHashCode()
    {
        unchecked // Overflow is fine, just wrap
        {
            var hash = 17;
            // Suitable nullity checks etc, of course :)
            hash = hash * 23 + Values.GetHashCode();
            hash = hash * 23 + ColumnNameToIndex.GetHashCode();

            return hash;
        }
    }
}
