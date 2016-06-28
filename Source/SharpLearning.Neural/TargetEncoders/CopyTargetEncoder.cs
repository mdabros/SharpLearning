using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Neural.TargetEncoders
{
    /// <summary>
    /// Copies the targets to a matrix of size [Target.Length, 1].
    /// Primary use is for regression.
    /// </summary>
    public sealed class CopyTargetEncoder : ITargetEncoder
    {
        /// <summary>
        /// Copies the targets to a matrix of size [Target.Length, 1].
        /// Primary use is for regression.
        /// </summary>
        /// <param name="targets"></param>
        /// <returns></returns>
        public Matrix<float> Encode(double[] targets)
        {
            var encodedTargets = Matrix<float>.Build.Dense(targets.Length, 1);
            for (int i = 0; i < targets.Length; i++)
            {
                encodedTargets[i, 0] = (float)targets[i];
            }

            return encodedTargets;
        }
    }
}
