using SharpLearning.Containers.Views;
using System.Linq;

namespace SharpLearning.Containers
{
    public static class ArrayExtensions
    {
        /// <summary>
        /// Gets the values from v based on indices
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="v"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public static T[] GetIndices<T>(this T[] v, int[] indices)
        {
            var result = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = v[indices[i]];
            }
            return result;
        }

        /// <summary>
        /// Converts am array of string to an array of floats
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static double[] AsF64(this string[] v)
        {
            return v.Select(s => FloatingPointConversion.ToF64(s)).ToArray();
        }

        /// <summary>
        /// Converts an array of doubles to an array of strings
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static string[] AsString(this double[] v)
        {
            return v.Select(s => FloatingPointConversion.ToString(s)).ToArray();
        }

        /// <summary>
        /// Converts an array of doubles to an array of ints
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static int[] AsInt32(this double[] v)
        {
            return v.Select(s => (int)s).ToArray();
        }

        /// <summary>
        /// Gets a pinned pointer to the double array
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static F64VectorPinnedPtr GetPinnedPointer(this double[] v)
        {
            return new F64VectorPinnedPtr(v);
        }
    }
}
