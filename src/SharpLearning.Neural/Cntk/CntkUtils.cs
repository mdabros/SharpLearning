using System;

namespace SharpLearning.Neural.Cntk
{
    public static class CntkUtils
    {
        public static void ConvertArray(double[] a1, float[] a2)
        {
            if (a1.Length != a2.Length)
            { throw new ArgumentException($"a1 length: {a1.Length} differs from {a2.Length}"); }

            for (int i = 0; i < a1.Length; i++)
            {
                a2[i] = (float)a1[i];
            }
        }
    }
}
