using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Containers
{
    public static class ArrayExtensions
    {
        public static T[] GetIndices<T>(this T[] v, int[] indices)
        {
            var result = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = v[indices[i]];
            }
            return result;
        }

        public static double[] AsF64(this string[] v)
        {
            return v.Select(s => FloatingPointConversion.ToF64(s)).ToArray();
        }

        public static string[] AsString(this double[] v)
        {
            return v.Select(s => FloatingPointConversion.ToString(s)).ToArray();
        }

    }
}
