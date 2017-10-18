using System;

namespace SharpLearning.Neural.Cntk
{
    public static class CntkUtils
    {
        public static void ConvertArray<T1, T2>(T1[] a1, T2[] a2)
        {
            if (a1.Length != a2.Length)
            { throw new ArgumentException($"a1 length: {a1.Length} differs from {a2.Length}"); }

            for (int i = 0; i < a1.Length; i++)
            {
                a2[i] = (T2)Convert.ChangeType(a1[i], typeof(T1));
            }
        }
    }
}
