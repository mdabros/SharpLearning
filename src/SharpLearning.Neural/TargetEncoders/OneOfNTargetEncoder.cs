using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace SharpLearning.Neural.TargetEncoders
{
    /// <summary>
    /// Encodes targets in a one-of-n structure. Target vector of with two classes [0, 1, 1, 0] becomes a matrix:
    /// 1 0
    /// 0 1
    /// 0 1
    /// 1 0
    /// Primary use is for classification
    /// </summary>
    public sealed class OneOfNTargetEncoder : ITargetEncoder
    {
        /// <summary>
        /// Encodes targets in a one-of-n structure. Target vector of with two classes [0, 1, 1, 0] becomes a matrix:
        /// 1 0
        /// 0 1
        /// 0 1
        /// 1 0
        /// Primary use is for classification
        /// </summary>
        /// <param name="targets"></param>
        /// <returns></returns>
        public Matrix<float> Encode(double[] targets)
        {
            var index = 0;
            var targetNameToTargetIndex = targets.Distinct().OrderBy(v => v)
                .ToDictionary(v => v, v => index++);

            var oneOfN = Matrix<float>.Build.Dense(targets.Length, targetNameToTargetIndex.Count);

            for (int i = 0; i < targets.Length; i++)
            {
                var target = targets[i];
                oneOfN[i, targetNameToTargetIndex[target]] = 1.0f;
            }

            return oneOfN;
        }
    }
}
