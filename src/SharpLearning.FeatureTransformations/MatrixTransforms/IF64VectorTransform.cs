namespace SharpLearning.FeatureTransformations.MatrixTransforms;

/// <summary>
/// Tranform for F64Vector
/// </summary>
public interface IF64VectorTransform
{
    /// <summary>
    /// Tranform for F64Vector
    /// </summary>
    /// <param name="vector"></param>
    /// <returns></returns>
    double[] Transform(double[] vector);

    /// <summary>
    /// Tranform for F64Vector
    /// </summary>
    /// <param name="vector"></param>
    /// <param name="output"></param>
    void Transform(double[] vector, double[] output);
}
