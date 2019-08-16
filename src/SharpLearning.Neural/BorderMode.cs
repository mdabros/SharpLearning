namespace SharpLearning.Neural
{
    /// <summary>
    /// Border mode for convolutional and pooling layers.
    /// </summary>
    public enum BorderMode
    {
        /// <summary>
        /// Pads with half the filter size on both sides. When stride is 1, this
        /// results in an output size equal to the input size.
        /// </summary>
        Same,

        /// <summary>
        /// Adds no padding. Only applies the kernel within the borders of the image. 
        /// </summary>
        Valid,

        /// <summary>
        /// Pads with one less than the filter size on both sides. This
        /// is equivalent to computing the convolution wherever the input and the
        /// filter overlap by at least one position.
        /// </summary>
        Full,

        /// <summary>
        /// No border mode defined.
        /// </summary>
        Undefined,
    }
}
