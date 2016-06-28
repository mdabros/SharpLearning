using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;

namespace SharpLearning.Neural.Optimizers
{
    /// <summary>
    /// Interface for neural net optimizers
    /// </summary>
    public interface INeuralNetOptimizer
    {
        /// <summary>
        /// Set the parameters for optimization
        /// </summary>
        /// <param name="coefs"></param>
        /// <param name="intercepts"></param>
        void SetParameters(List<Matrix<float>> coefs, List<Vector<float>> intercepts);

        /// <summary>
        /// Updates the parameters
        /// </summary>
        /// <param name="coefGrad"></param>
        /// <param name="interceptGrad"></param>
        void UpdateParameters(List<Matrix<float>> coefGrad, List<Vector<float>> interceptGrad);

        /// <summary>
        /// Complete necesarry updates when an iteration ends
        /// </summary>
        /// <param name="samples"></param>
        void IterationEnds(int samples);

        /// <summary>
        /// Stop the optimization if necesarry
        /// </summary>
        /// <returns></returns>
        bool TriggerStopping();
    }
}
