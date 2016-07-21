using SharpLearning.Containers.Matrices;

namespace SharpLearning.FeatureTransformations.MatrixTransforms
{
    /// <summary>
    /// Tranform for F64Matrix
    /// </summary>
    interface IF64MatrixTransform
    {
        /// <summary>
        /// Tranform for F64Matrix
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        F64Matrix Transform(F64Matrix matrix);

        /// <summary>
        /// Tranform for F64Matrix
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="output"></param>
        void Transform(F64Matrix matrix, F64Matrix output);
    }
}
