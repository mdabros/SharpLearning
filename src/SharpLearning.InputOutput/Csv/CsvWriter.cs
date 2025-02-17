﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpLearning.InputOutput.Csv;

public class CsvWriter
{
    readonly Func<TextWriter> m_writer;
    readonly char m_separator;

    /// <summary>
    /// Creates a CsvWriter
    /// </summary>
    /// <param name="writer"></param>
    /// <param name="separator"></param>
    public CsvWriter(Func<TextWriter> writer, char separator = CsvParser.DefaultDelimiter)
    {
        m_writer = writer ?? throw new ArgumentNullException(nameof(writer));
        m_separator = separator;
    }

    /// <summary>
    /// Writes the CsvRows to stream
    /// </summary>
    /// <param name="rows">the rows to write</param>
    /// <param name="writeHeader">True and a header is added to the stream, false and the header is omittet</param>
    public void Write(IEnumerable<CsvRow> rows, bool writeHeader = true)
    {
        using var writer = m_writer();
        var rowsList = rows.ToList();
        if (writeHeader)
        {
            var headerValues = rowsList[0].ColumnNameToIndex
                                     .OrderBy(kvp => kvp.Value)
                                     .Select(kvp => kvp.Key);

            var headerLine = CreateHeader(headerValues);
            writer.Write(headerLine);
        }

        foreach (var row in rowsList)
        {
            writer.WriteLine();
            WriteColumns(row.Values, writer);
        }
    }

    string CreateHeader(IEnumerable<string> headerValues)
    {
        var headerLine = string.Empty;
        using (var enumerator = headerValues.GetEnumerator())
        {
            var moveNext = enumerator.MoveNext();
            while (moveNext)
            {
                headerLine += enumerator.Current;

                moveNext = enumerator.MoveNext();

                if (moveNext)
                {
                    headerLine += m_separator;
                }
            }
        }

        return headerLine;
    }

    void WriteColumns(string[] values, TextWriter writer)
    {
        var enumerator = values.GetEnumerator();

        var moveNext = enumerator.MoveNext();
        while (moveNext)
        {
            writer.Write(enumerator.Current);

            moveNext = enumerator.MoveNext();

            if (moveNext)
            {
                writer.Write(m_separator);
            }
        }
    }
}
