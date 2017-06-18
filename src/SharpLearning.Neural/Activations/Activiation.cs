namespace SharpLearning.Neural.Activations
{
    /// <summary>
    /// Activation type for neural net. 
    /// </summary>
    public enum Activation
    {
        /// <summary>
        /// No activation.
        /// </summary>
        Undefined,

        /// <summary>
        /// Relu activation.
        /// </summary>
        Relu,

        /// <summary>
        /// Sigmoid activation.
        /// </summary>
        Sigmoid,

        /// <summary>
        /// SoftMax activation.
        /// </summary>
        SoftMax,

        /// <summary>
        /// MeanSquareError activation.
        /// </summary>
        MeanSquareError,

        /// <summary>
        /// Svm activation (also caleed hinge).
        /// </summary>
        Svm,
    }
}
