using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using SharpLearning.Containers;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations.CsvRowTransforms
{
    /// <summary>
    /// Transform a data string into numerical features that can be presented to a machine learning algorithm.
    /// The following features are extracted:
    /// Year, Month, DayOfMonth, DayOfWeek, HourOfDay, TotalDays, TotalHours.
    /// The TotalDays and TotalHours are calculated from a provided start date or if none provided: 1970-1-1.
    /// </summary>
    [Serializable]
    public sealed class DateTimeFeatureTransformer : ICsvRowTransformer
    {
        readonly DateTime m_startDate;
        readonly string m_dateTimeColumn;

        /// <summary>
        /// Transform a data string into numerical features that can be presented to a machine learning algorithm.
        /// The following features are extracted:
        /// Year, Month, DayOfMonth, DayOfWeek, HourOfDay, TotalDays, TotalHours.
        /// The TotalDays and TotalHours are calculated from a provided start date or if none provided: 1970-1-1.
        /// </summary>
        /// <param name="dateTimeColumn">column name of datetime column</param>
        public DateTimeFeatureTransformer(string dateTimeColumn)
            : this(dateTimeColumn, new DateTime(1970, 1, 1, 0, 0, 0))
        {
        }

        /// <summary>
        /// Transform a data string into numerical features that can be presented to a machine learning algorithm.
        /// The following features are extracted:
        /// Year, Month, DayOfMonth, DayOfWeek, HourOfDay, TotalDays, TotalHours.
        /// The TotalDays and TotalHours are calculated from a provided start date or if none provided: 1970-1-1.
        /// </summary>
        /// <param name="dateTimeColumn">column name of datetime column</param>
        /// <param name="startDate"></param>
        public DateTimeFeatureTransformer(string dateTimeColumn, DateTime startDate)
        {
            if (startDate == null) { throw new ArgumentException("startDate"); }
            m_dateTimeColumn = dateTimeColumn;
            m_startDate = startDate;
        }

        /// <summary>
        /// 
        /// </summary>
        public string[] FeatureNames
        {
            get => new[] { "Year", "Month", "WeekOfYear", "DayOfMonth", "DayOfWeek", "HourOfDay", "TotalDays", "TotalHours" };
        }

        /// <summary>
        /// Transform a data string into numerical features that can be presented to a machine learning algorithm.
        /// The following features are extracted:
        /// Year, Month, DayOfMonth, DayOfWeek, HourOfDay, TotalDays, TotalHours.
        /// The TotalDays and TotalHours are calculated from a provided start date or if none provided: 1970-1-1.
        /// </summary>
        /// <param name="rows"></param>
        /// <returns></returns>
        public IEnumerable<CsvRow> Transform(IEnumerable<CsvRow> rows)
        {
            var newColumnNameToIndex = rows.First().ColumnNameToIndex.ToDictionary(v => v.Key, v => v.Value);
            var index = newColumnNameToIndex.Count;

            foreach (var name in FeatureNames)
            {
                var newName = CreateKey(name, newColumnNameToIndex);
                newColumnNameToIndex.Add(newName, index++);
            }

            foreach (var row in rows)
            {
                var dateTime = DateTime.Parse(row.GetValue(m_dateTimeColumn));

                var timeValues = CreateDateTimeFeatures(dateTime);
                var newValues = new string[row.Values.Length + timeValues.Length];

                row.Values.CopyTo(newValues, 0);
                timeValues.CopyTo(newValues, row.Values.Length);

                yield return new CsvRow(newColumnNameToIndex, newValues);
            }
        }

        string[] CreateDateTimeFeatures(DateTime dateTime)
        {
            var year = dateTime.Year;
            var month = dateTime.Month;
            var dayOfMonth = dateTime.Day;
            var dayOfWeek = (int)dateTime.DayOfWeek;
            var hours = dateTime.TimeOfDay.Hours;
            var totalDays = Math.Round(dateTime.Subtract(m_startDate).TotalDays, 1);
            var totalhours = Math.Round(dateTime.Subtract(m_startDate).TotalHours, 1);

            var dfi = DateTimeFormatInfo.CurrentInfo;
            var cal = dfi.Calendar;
            var weekOfYear = cal.GetWeekOfYear(dateTime, dfi.CalendarWeekRule, dfi.FirstDayOfWeek);

            var timeValues = new string[]
            {
                FloatingPointConversion.ToString(year),
                FloatingPointConversion.ToString(month),
                FloatingPointConversion.ToString(weekOfYear),
                FloatingPointConversion.ToString(dayOfMonth),
                FloatingPointConversion.ToString(dayOfWeek),
                FloatingPointConversion.ToString(hours),
                FloatingPointConversion.ToString(totalDays),
                FloatingPointConversion.ToString(totalhours),

            };
            return timeValues;
        }

        string CreateKey(string key, Dictionary<string, int> columnNameToIndex)
        {
            if (!columnNameToIndex.ContainsKey(key))
            {
                return key;
            }
            else
            {
                var index = 1;
                var newKey = key + "_" + index;
                while (columnNameToIndex.ContainsKey(newKey))
                {
                    index++;
                    newKey = key + "_" + index;
                }
                return newKey;
            }
        }
    }
}
