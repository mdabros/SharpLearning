using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpLearning.InputOutput.Csv
{
    public class CsvWriter : IDisposable
    {
        TextWriter m_writer;
        readonly char m_separator;

        /// <summary>
        /// Creates a CsvWriter
        /// </summary>
        /// <param name="writer"></param>
        /// <param name="separator"></param>
        public CsvWriter(TextWriter writer, char separator = CsvParser.DefaultDelimiter)
        {
            if (writer == null) { throw new ArgumentException("writer"); }
            m_writer = writer;
            m_separator = separator;
        }

        /// <summary>
        /// Writes the CsvRows to file
        /// </summary>
        /// <param name="rows"></param>
        public void Write(IEnumerable<CsvRow> rows)
        {
            var headerValues = rows.First().ColumnNameToIndex
                                     .OrderBy(kvp => kvp.Value)
                                     .Select(kvp => kvp.Key);

            var headerLine = CreateHeader(headerValues);
            m_writer.Write(headerLine);

            foreach (var row in rows)
            {
                m_writer.WriteLine();
                WriteColumns(row.Values, m_writer);
            }
        }

        string CreateHeader(IEnumerable<string> headerValues)
        {
            var headerLine = string.Empty;
            using (var enumerator = headerValues.GetEnumerator())
            {
                bool moveNext = enumerator.MoveNext();
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

            bool moveNext = enumerator.MoveNext();
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

        private volatile bool m_disposed = false;
        private void DisposeManagedResources()
        {
            if (m_writer != null)
            {
                IDisposable disposable = m_writer;
                m_writer = null;
                disposable.Dispose();
            }
        }

        public void Dispose()
        {
            Dispose(true);
        }

        protected void Dispose(bool disposing)
        {
            if (!m_disposed)
            {
                if (disposing)
                {
                    DisposeManagedResources();
                }
            }
            m_disposed = true;
        }
    }
}
