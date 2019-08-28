using System.Linq;

namespace SharpLearning.DataSource
{
    public static class Converters
    {

        /// <summary>
        /// Converts bytes to float.
        /// </summary>
        /// <param name="bytes"></param>
        /// <returns></returns>
        public static float[] ToFloat(byte[] bytes)
        {
            return bytes.Select(v => (float)v).ToArray();
        }

        /// <summary>
        /// Converts bytes to double.
        /// </summary>
        /// <param name="bytes"></param>
        /// <returns></returns>
        public static double[] ToDouble(byte[] bytes)
        {
            return bytes.Select(v => (double)v).ToArray();
        }
    }
}
