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
            m_getReader = reader ?? throw new ArgumentException("reader");
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
            if(!m_hasHeader)
            {
                throw new ArgumentException("CsvParser configured to use no header." + 
                    " Column names cannot be selected in this made");
            }

            using (var reader = m_getReader())
            {
                var headerLine = reader.ReadLine();
                var columnNameToIndex = TrimSplitLineTrimColumnsToDictionary(headerLine);
                var columnNames = columnNameToIndex.Keys.Where(name => selectColumnNames(name))
                    .ToArray();

                var indices = columnNameToIndex.GetValues(columnNames);
                var subColumnNameToIndex = Enumerable.Range(0, indices.Length)
                    .ToDictionary(index => columnNames[index]);

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
            if (!m_hasHeader)
            {
                throw new ArgumentException("CsvParser configured to use no header." + 
                    "Column names cannot be selected in this made");
            }

            using (var reader = m_getReader())
            {
                var headerLine = reader.ReadLine();
                var columnNameToIndex = TrimSplitLineTrimColumnsToDictionary(headerLine);
                var indices = columnNameToIndex.GetValues(columnNames);
                var subColumnNameToIndex = Enumerable.Range(0, indices.Length)
                    .ToDictionary(index => columnNames[index]);

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
            if(m_hasHeader)
            {
                return EnumerateRowsHeader();
            }
            else
            {
                return EnumerateRowsNoHeader();
            }
        }

        IEnumerable<CsvRow> EnumerateRowsHeader()
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
        
        IEnumerable<CsvRow> EnumerateRowsNoHeader()
        {
            var columnNameToIndex = CreateHeaderForCsvFileWithout();

            using (var reader = m_getReader())
            {
                string line = null;
                while ((line = reader.ReadLine()) != null)
                {
                    var lineSplit = Split(line);
                    yield return new CsvRow(columnNameToIndex, lineSplit);
                }
            }
        }

        Dictionary<string, int> CreateHeaderForCsvFileWithout()
        {
            Dictionary<string, int> columnNameToIndex = new Dictionary<string, int>();

            // create header for csv file without header.
            using (var reader = m_getReader())
            {
                var line = reader.ReadLine();
                var splitLine = Split(line);

                for (int i = 0; i < splitLine.Length; i++)
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
            string[] split = null;

            if (m_quoteInclosedColumns)
            {
                split = SplitText(line, m_separator);
            }
            else
            {
                split = line.Split(m_separator);
            }

            for (int i = 0; i < split.Length; i++)
            {
                split[i] = split[i].Trim('"');
            }
            return split;
        }

        string[] Split(string line, int[] indices)
        {
            string[] splitAll = null;

            if(m_quoteInclosedColumns)
            {
                splitAll = SplitText(line, m_separator);
            }
            else
            {
                splitAll = line.Split(m_separator);
            }
            
            var split = new string[indices.Length];

            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                split[i] = splitAll[index].Trim('"');
            }

            return split;
        }

        string[] SplitText(string csvText, char separator)
        {
            List<string> tokens = new List<string>();

            int last = -1;
            int current = 0;
            bool inText = false;

            while (current < csvText.Length)
            {
                var token = csvText[current]; 
                
                if(token == '"')
                {
                    inText = !inText;
                }
                else if(token == separator)
                {
                    if (!inText)
                    {
                        tokens.Add(csvText.Substring(last + 1, (current - last)).Trim(' ', separator));
                        last = current;
                    }
                }

                current++;
            }

            if (last != csvText.Length - 1)
            {
                tokens.Add(csvText.Substring(last + 1).Trim());
            }

            return tokens.ToArray();
        }
    }
}
