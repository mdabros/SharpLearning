using System.Collections.Generic;
using CNTK;

namespace CntkCatalyst
{
    /// <summary>
    /// 
    /// </summary>
    public static class CntkUtils
    {
        internal static ParameterVector CreateParameterVector(IList<Parameter> input)
        {
            ParameterVector inputVector = new ParameterVector();
            foreach (var element in input)
            {
                inputVector.Add(element);
            }
            return inputVector;
        }

        internal static BoolVector CreateFilledBoolVector(int size, bool fill)
        {
            var boolVector = new BoolVector(size);
            for (int i = 0; i < size; i++)
            {
                boolVector.Add(fill);
            }
            return boolVector;
        }
    }
}
