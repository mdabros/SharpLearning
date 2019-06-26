using System;
using System.Globalization;

namespace SharpLearning.Containers
{
    /// <summary>
    /// 
    /// </summary>
    public static class FloatingPointConversion
    {
        /// <summary>
        /// 
        /// </summary>
        public const string DefaultFormat = "R";

        /// <summary>
        /// 
        /// </summary>
        public static readonly NumberFormatInfo nfi = new NumberFormatInfo();

        /// <summary>
        /// Default NumberStyle is Any.
        /// </summary>
        public static readonly NumberStyles NumberStyle = NumberStyles.Any;

        /// <summary>
        /// Default format for outputting double values to string.
        /// </summary> 
        public static string ToString(double value)
        {
            return value.ToString(DefaultFormat, nfi);
        }

        /// <summary>
        /// Default format for converting string values to double
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double ToF64(string value)
        {
            return ToF64(value, ParseAnyNumberStyle);
        }

        /// <summary>
        /// Allows for custom conversion of string to double.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="converter"></param>
        /// <returns></returns>
        public static double ToF64(string value, Converter<string, double> converter)
        {
            return converter(value);
        }

        static double ParseAnyNumberStyle(string value)
        {
            if (double.TryParse(value, NumberStyle, nfi, out double result))
            {
                return result;
            }
            else
            {
                throw new ArgumentException($"Unable to parse \"{ value }\" to double");
            }
        }
    }
}
