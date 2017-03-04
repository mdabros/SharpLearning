using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public static class Utils
    {
        ///// <summary>
        ///// SIMD multiply
        ///// </summary>
        ///// <param name="v1"></param>
        ///// <param name="v2"></param>
        ///// <returns></returns>
        //public static float DotSimd(float[] v1, float[] v2)
        //{
        //    var simdLength = Vector<float>.Count;
        //    var i = 0;

        //    var result = 0f;

        //    for (i = 0; i <= v1.Length - simdLength; i += simdLength)
        //    {
        //        var va = new Vector<float>(v1, i);
        //        var vb = new Vector<float>(v2, i);
        //        result += Vector.Dot(va, vb);
        //    }

        //    for (; i < v1.Length; ++i)
        //    {
        //        result = v1[i] * v2[i];
        //    }

        //    return result;
        //}

        /// <summary>
        /// 
        /// </summary>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <returns></returns>
        public static float Dot(float[] v1, float[] v2)
        {
            var result = 0f;

            for (int i = 0; i < v1.Length; i++)
            {
                result += v1[i] * v2[i];
            }

            return result;
        }
    }
}
