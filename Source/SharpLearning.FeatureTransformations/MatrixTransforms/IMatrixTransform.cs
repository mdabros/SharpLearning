using SharpLearning.Containers.Matrices;

namespace SharpLearning.FeatureTransformations.MatrixTransforms
{
    interface IF64MatrixTransform
    {
        F64Matrix Transform(F64Matrix matrix);
        void Transform(F64Matrix matrix, F64Matrix output);
    }
}
