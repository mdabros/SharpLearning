using System;

namespace SharpLearning.Neural.Layers
{
    /// <summary>
    /// Hidden layer for neural network learners.
    /// </summary>
    public struct HiddenLayer
    {
        /// <summary>
        /// Number of hidden units or neurons
        /// </summary>
        public readonly int Units;

        /// <summary>
        /// Dropout percentage.
        /// </summary>
        public readonly double DropOut;

        /// <summary>
        /// Hidden layer for neural network learners.
        /// </summary>
        /// <param name="units">Number of hidden units or neurons in the layer</param>
        /// <param name="dropOut">Dropout percentage. The percentage of units randomly omitted during training.
        /// This is a reguralizatin methods for reducing overfitting. Recommended value is 0.5 and range should be between 0.2 and 0.8.
        /// Default (0.0). https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf</param>
        public HiddenLayer(int units, double dropOut = 0.0)
        {
            if (units < 1) { throw new ArgumentException("HiddenLayer must have at least 1 hidden unit"); }
            if (dropOut < 0.0 || dropOut >= 1.0) { throw new ArgumentException("Dropout must be below 1.0 and at least 0.0"); }
            Units = units;
            DropOut = dropOut;
        }

        /// <summary>
        /// Creates a new hidden layer
        /// </summary>
        /// <param name="units">Number of hidden units or neurons in the layer</param>
        /// <param name="dropOut">Dropout percentage. The percentage of units randomly omitted during training.
        /// This is a reguralizatin methods for reducing overfitting. Recommended value is 0.5 and range should be between 0.2 and 0.8.
        /// Default (0.0). https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf</param>
        /// <returns></returns>
        public static HiddenLayer New(int units, double dropOut = 0.0)
        {
            return new HiddenLayer(units, dropOut);
        }
    }
}
