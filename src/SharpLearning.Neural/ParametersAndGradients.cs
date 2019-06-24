namespace SharpLearning.Neural
{
    /// <summary>
    /// 
    /// </summary>
    public class ParametersAndGradients
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly float[] Parameters;

        /// <summary>
        /// 
        /// </summary>
        public readonly float[] Gradients;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parameters"></param>
        /// <param name="gradients"></param>
        public ParametersAndGradients(float[] parameters, float[] gradients)
        {
            Parameters = parameters;
            Gradients = gradients;
        }
    }
}
