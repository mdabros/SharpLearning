namespace SharpLearning.Neural.Layers
{
    /// <summary>
    /// Interface implemented by layers that support batch normalization.
    /// </summary>
    public interface IBatchNormalizable
    {
        /// <summary>
        /// Does the layer use batch normalization
        /// </summary>
        bool BatchNormalization { get; set; }
    }
}
