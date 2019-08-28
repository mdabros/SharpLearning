using System.Collections.Generic;
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

        /// <summary>
        /// Transform value to onehot vector.
        /// </summary>
        /// <param name="value"></param>
        /// <param name="valueToOneHotIndex"></param>
        /// <returns></returns>
        public static float[] ToOneHot(float value, IReadOnlyDictionary<float, int> valueToOneHotIndex)
        {
            var oneHotVector = new float[valueToOneHotIndex.Count];
            var oneHotIndex = valueToOneHotIndex[value];
            oneHotVector[oneHotIndex] = 1;
            return oneHotVector;
        }
    }
}
