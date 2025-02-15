using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpLearning.InputOutput.Csv;

/// <summary>
/// CsvParser
/// </summary>
public sealed class CsvParser
{
    /// <summary>
    /// Default delimiter
    /// </summary>
    public const char DefaultDelimiter = ';';

    readonly Func<TextReader> m_getReader;
    readonly char m_separator;
    readonly bool m_quoteInclosedColumns;
    readonly bool m_hasHeader;

    /// <summary>
    /// Creates an instance of the CsvParser
    /// </summary>
    /// <param name="reader"></param>
    /// <param name="separator"></param>
    /// <param name="quoteInclosedColumns"></param>
    /// <param name="hasHeader"></param>
    public CsvParser(Func<TextReader> reader,
        char separator = DefaultDelimiter,
        bool quoteInclosedColumns = false,
        bool hasHeader = true)
    {
        m_getReader = reader ?? throw new ArgumentNullException(nameof(reader));
        m_separator = separator;
        m_quoteInclosedColumns = quoteInclosedColumns;
        m_hasHeader = hasHeader;
    }

    /// <summary>
    /// Enumerates rows with column names fulfilling the selection function
    /// </summary>
    /// <param name="selectColumnNames"></param>
    /// <returns></returns>
    public IEnumerable<CsvRow> EnumerateRows(Func<string, bool> selectColumnNames)
    {
        if (!m_hasHeader)
        {
            throw new ArgumentException("CsvParser configured to use no header."
                + " Column names cannot be selected in this mode");
        }

        var (columnNameToIndex, reader) = ReadHeader();
        var columnNames = columnNameToIndex.Keys.Where(selectColumnNames).ToArray();

        if (columnNames.Length == 0)
        {
            throw new InvalidOperationException("Column names has length 0.");
        }

        var indices = columnNameToIndex.GetValues(columnNames);
        var subColumnNameToIndex = Enumerable.Range(0, indices.Length)
            .ToDictionary(index => columnNames[index]);

        return EnumerateRows(reader, subColumnNameToIndex, indices);
    }

    /// <summary>
    /// Enumerates rows with the specified column names
    /// </summary>
    /// <param name="columnNames"></param>
    /// <returns></returns>
    public IEnumerable<CsvRow> EnumerateRows(params string[] columnNames)
    {
        if (!m_hasHeader)
        {
            throw new ArgumentException("CsvParser configured to use no header."
                + " Column names cannot be selected in this mode");
        }

        var (columnNameToIndex, reader) = ReadHeader();
        var indices = columnNameToIndex.GetValues(columnNames);
        var subColumnNameToIndex = Enumerable.Range(0, indices.Length)
            .ToDictionary(index => columnNames[index]);

        return EnumerateRows(reader, subColumnNameToIndex, indices);
    }

    /// <summary>
    /// Enumerates the row of all columns in the csv file
    /// </summary>
    /// <returns></returns>
    public IEnumerable<CsvRow> EnumerateRows()
    {
        return m_hasHeader ? EnumerateRowsHeader() : EnumerateRowsNoHeader();
    }

    IEnumerable<CsvRow> EnumerateRowsHeader()
    {
        var (columnNameToIndex, reader) = ReadHeader();
        return EnumerateRows(reader, columnNameToIndex);
    }

    IEnumerable<CsvRow> EnumerateRowsNoHeader()
    {
        var columnNameToIndex = CreateHeaderForCsvFileWithout();
        using var reader = m_getReader();
        return EnumerateRows(reader, columnNameToIndex);
    }

    (Dictionary<string, int> columnNameToIndex, TextReader reader) ReadHeader()
    {
        var reader = m_getReader();
        var headerLine = reader.ReadLine() ?? throw new InvalidOperationException(
            "The CSV file is empty or the reader returned null.");
        var columnNameToIndex = TrimSplitLineTrimColumnsToDictionary(headerLine);
        return (columnNameToIndex, reader);
    }

    IEnumerable<CsvRow> EnumerateRows(TextReader reader,
        Dictionary<string, int> columnNameToIndex, int[] indices = null)
    {
        string line;
        while ((line = reader.ReadLine()) != null)
        {
            var lineSplit = indices == null
                ? Split(line)
                : Split(line, indices);

            yield return new CsvRow(columnNameToIndex, lineSplit);
        }
    }

    Dictionary<string, int> CreateHeaderForCsvFileWithout()
    {
        var columnNameToIndex = new Dictionary<string, int>();

        // create header for csv file without header.
        using (var reader = m_getReader())
        {
            var line = reader.ReadLine() ?? throw new InvalidOperationException(
                "The CSV file is empty or the reader returned null.");
            var splitLine = Split(line);

            for (var i = 0; i < splitLine.Length; i++)
            {
                columnNameToIndex.Add(i.ToString(), i);
            }
        }

        return columnNameToIndex;
    }

    Dictionary<string, int> TrimSplitLineTrimColumnsToDictionary(string line)
    {
        var dictionary = new Dictionary<string, int>();
        var lineSplit = Split(line);
        for (var index = 0; index < lineSplit.Length; index++)
        {
            var trimmedItem = lineSplit[index].Trim();
            dictionary.Add(trimmedItem, index);
        }
        return dictionary;
    }

    string[] Split(string line)
    {
        var split = m_quoteInclosedColumns
            ? SplitText(line, m_separator)
            : line.Split(m_separator);

        for (var i = 0; i < split.Length; i++)
        {
            split[i] = split[i].Trim('"');
        }
        return split;
    }

    string[] Split(string line, int[] indices)
    {
        var splitAll = m_quoteInclosedColumns
            ? SplitText(line, m_separator)
            : line.Split(m_separator);
        var split = new string[indices.Length];

        for (var i = 0; i < indices.Length; i++)
        {
            var index = indices[i];
            split[i] = splitAll[index].Trim('"');
        }

        return split;
    }

    static string[] SplitText(string csvText, char separator)
    {
        var tokens = new List<string>();
        var last = -1;
        var inText = false;

        for (var current = 0; current < csvText.Length; current++)
        {
            var token = csvText[current];

            if (token == '"')
            {
                inText = !inText;
            }
            else if (token == separator && !inText)
            {
                tokens.Add(csvText.Substring(last + 1, current - last).Trim(' ', separator));
                last = current;
            }
        }

        if (last != csvText.Length - 1)
        {
            tokens.Add(csvText.Substring(last + 1).Trim());
        }

        return tokens.ToArray();
    }
}
