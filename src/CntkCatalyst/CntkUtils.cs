using System.Collections.Generic;
using System.Linq;
using CNTK;

namespace CntkCatalyst
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
        public static ParameterVector AsParameterVector(IList<Parameter> input)
        {
            ParameterVector inputVector = new ParameterVector();
            foreach (var element in input)
            {
                inputVector.Add(element);
            }
            return inputVector;
        }

        /// <summary>
        /// Encodes targets in a one-of-n structure. Target vector of with two classes [0, 1, 1, 0] becomes a matrix:
        /// 1 0
        /// 0 1
        /// 0 1
        /// 1 0
        /// Primary use is for classification
        /// </summary>
        public static float[] EncodeOneHot(this float[] targets)
        {
            var index = 0;
            var targetNameToTargetIndex = targets.Distinct().OrderBy(v => v)
                .ToDictionary(v => v, v => index++);

            var distinctTargets = targetNameToTargetIndex.Count;
            var oneHot = new float[targets.Length * distinctTargets];

            for (int i = 0; i < targets.Length; i++)
            {
                var target = targets[i];
                var targetIndex = targetNameToTargetIndex[target];
                var oneHotIndex = i * distinctTargets + targetIndex;
                oneHot[oneHotIndex] = 1.0f;
            }

            return oneHot;
        }
    }
}
