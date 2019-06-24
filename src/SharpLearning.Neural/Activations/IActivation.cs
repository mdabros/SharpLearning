namespace SharpLearning.Neural.Activations
{
    /// <summary>
    /// Neural net activiation interface
    /// </summary>
    public interface IActivation
    {
        /// <summary>
        /// Calcualtes the activation and stores the result in x
        /// </summary>
        /// <param name="x"></param>
        void Activation(float[] x);

        /// <summary>
        /// Calculates the derivative and stores the result in output
        /// </summary>
        /// <param name="x"></param>
        /// <param name="output"></param>
        void Derivative(float[] x, float[] output);
    }
}
