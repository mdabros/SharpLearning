using System;
using System.Collections.Generic;
using CNTK;

namespace CntkExtensions
{
    /// <summary>
    /// 
    /// </summary>
    public static class CntkUtils
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public  static ParameterVector AsParameterVector(IList<Parameter> input)
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
