using System.Globalization;

namespace SharpLearning.Containers
{
    public static class FloatingPointConversion
    {
        public const string DefaultFormat = "R";

        /// <summary>
        /// Default format for outputting double values to string.
        /// </summary> 
        public static string ToString(double value)
        {
            var nfi = new NumberFormatInfo();
            return value.ToString(DefaultFormat, nfi);
        }

        /// <summary>
        /// Default format for converting string values to double
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double ToF64(string value)
        {
            var nfi = new NumberFormatInfo();
            return double.Parse(value, nfi);
        }
    }
}
