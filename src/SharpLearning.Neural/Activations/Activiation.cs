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
        /// SoftMax activation.
        /// </summary>
        SoftMax,

        /// <summary>
        /// MeanSquareError activation.
        /// </summary>
        MeanSquareError,
    }
}
