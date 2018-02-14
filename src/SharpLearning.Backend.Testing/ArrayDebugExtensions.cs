using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLearning.Backend.Testing
{
    public static class ArrayDebugExtensions
    {
        public static string ToDebugText(this Array a)
        {
            var sb = new StringBuilder((a.Rank * 8) + (a.Length * 32));
            var indices = new int[a.Rank];
            RecurseToDebugText(a, 0, indices, sb);
            return sb.ToString();
        }

        private static void RecurseToDebugText(Array a, int dim, int[] indices, StringBuilder sb)
        {
            var l = a.GetLength(dim);
            for (int i = 0; i < l; i++)
            {
                indices[dim] = i;
                if (dim < (a.Rank - 1))
                {
                    RecurseToDebugText(a, dim + 1, indices, sb);
                }
                else
                {
                    var v = a.GetValue(indices);
                    sb.Append("[");
                    sb.Append(string.Join(",", indices));
                    sb.Append("]=");
                    sb.Append(v);
                    sb.AppendLine("");
                }
            }
        }
    }
}
