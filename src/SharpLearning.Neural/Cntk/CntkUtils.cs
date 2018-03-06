using System;
using System.Collections.Generic;
using CNTK;

namespace SharpLearning.Neural.Cntk
{
    /// <summary>
    /// 
    /// </summary>
    public static class CntkUtils
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a1"></param>
        /// <param name="a2"></param>
        public static void ConvertArray(double[] a1, float[] a2)
        {
            if (a1.Length != a2.Length)
            { throw new ArgumentException($"a1 length: {a1.Length} differs from {a2.Length}"); }

            for (int i = 0; i < a1.Length; i++)
            {
                a2[i] = (float)a1[i];
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        internal static ParameterVector AsParameterVector(IList<Parameter> input)
        {
            ParameterVector inputVector = new ParameterVector();
            foreach (var element in input)
            {
                inputVector.Add(element);
            }
            return inputVector;
        }
    }
}
