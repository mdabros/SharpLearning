using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpLearning.InputOutput.Csv
{
    /// <summary>
    /// CsvParser 
    /// </summary>
    public sealed class CsvParser
    {
        public const char DefaultDelimiter = ';';

        readonly Func<TextReader> m_getReader;
        readonly char m_separator;

        /// <summary>
        /// Creates an instance of the CsvParser
        /// </summary>
        /// <param name="reader"></param>
        /// <param name="separator"></param>
        public CsvParser(Func<TextReader> reader, char separator = DefaultDelimiter)
        {
            if (reader == null) { throw new ArgumentException("reader"); }
            m_getReader = reader;
            m_separator = separator;
        }

        /// <summary>
        /// Enumerates rows with column names fulfilling the selection func 
        /// </summary>
        /// <param name="selectColumnNames"></param>
        /// <returns></returns>
        public IEnumerable<CsvRow> EnumerateRows(Func<string, bool> selectColumnNames)
        {
            using (var reader = m_getReader())
            {
                var headerLine = reader.ReadLine();
                var columnNameToIndex = TrimSplitLineTrimColumnsToDictionary(headerLine);
                var columnNames = columnNameToIndex.Keys.Where(name => selectColumnNames(name)).ToArray();
                var indices = columnNameToIndex.GetValues(columnNames);
                var subColumnNameToIndex = Enumerable.Range(0, indices.Length).ToDictionary(index => columnNames[index]);

                string line = null;
                while ((line = reader.ReadLine()) != null)
                {
                    var lineSplit = Split(line, indices);
                    yield return new CsvRow(subColumnNameToIndex, lineSplit);
                }
            }
        }

        /// <summary>
        /// Enumerates rows with the specified column names
        /// </summary>
        /// <param name="columnNames"></param>
        /// <returns></returns>
        public IEnumerable<CsvRow> EnumerateRows(params string[] columnNames)
        {
            using (var reader = m_getReader())
            {
                var headerLine = reader.ReadLine();
                var columnNameToIndex = TrimSplitLineTrimColumnsToDictionary(headerLine);
                var indices = columnNameToIndex.GetValues(columnNames);
                var subColumnNameToIndex = Enumerable.Range(0, indices.Length).ToDictionary(index => columnNames[index]);

                string line = null;
                while ((line = reader.ReadLine()) != null)
                {
                    var lineSplit = Split(line, indices);
                    yield return new CsvRow(subColumnNameToIndex, lineSplit);
                }
            }
        }

        /// <summary>
        /// Enumerates the row of all columns in the csv file 
        /// </summary>
        /// <returns></returns>
        public IEnumerable<CsvRow> EnumerateRows()
        {
            using (var reader = m_getReader())
            {
                var headerLine = reader.ReadLine();
                var columnNameToIndex = TrimSplitLineTrimColumnsToDictionary(headerLine);

                string line = null;
                while ((line = reader.ReadLine()) != null)
                {
                    var lineSplit = Split(line);
                    yield return new CsvRow(columnNameToIndex, lineSplit);
                }
            }
        }

        Dictionary<string, int> TrimSplitLineTrimColumnsToDictionary(string line)
        {
            var dictionary = new Dictionary<string, int>();
            var lineSplit = Split(line);
            var index = 0;
            foreach (var item in lineSplit)
            {
                var trimmedItem = item.Trim();
                dictionary.Add(trimmedItem, index);
                index++;
            }
            return dictionary;
        }

        string[] Split(string line)
        {
            var split = line.Split(m_separator);
            for (int i = 0; i < split.Length; i++)
            {
                split[i] = split[i].Trim('"');
            }
            return split;
        }

        string[] Split(string line, int[] indices)
        {
            var splitAll = line.Split(m_separator);
            var split = new string[indices.Length];

            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                split[i] = splitAll[index].Trim('"');
            }

            return split;
        }
    }
}
