using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Neural.Initializations;
using SharpLearning.Neural.Providers.DotNetOp;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class DropoutLayer : ILayerNew
    {

        /// <summary>
        /// Dropout percentage.
        /// </summary>
        readonly double DropOut;

        Random m_random;

        Variable DropoutMask;

        /// <summary>
        /// Dropout layer for neural network learners.
        /// </summary>
        /// <param name="dropOut">Dropout percentage. The percentage of units randomly omitted during training.
        /// This is a reguralizatin methods for reducing overfitting. Recommended value is 0.5 and range should be between 0.2 and 0.8.
        /// Default (0.0). https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf</param>
        public DropoutLayer(double dropOut = 0.0)
        {
            if (dropOut < 0.0 || dropOut >= 1.0) { throw new ArgumentException("Dropout must be below 1.0 and at least 0.0"); }
            DropOut = dropOut;
        }

        /// <summary>
        /// 
        /// </summary>
        public Variable Output { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public Variable Input { get; set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="storage"></param>
        public void Backward(NeuralNetStorage storage)
        {
            Dropout.Backward(Input, DropoutMask, Output, storage);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="storage"></param>
        /// <param name="copyStorage"></param>
        /// <param name="layers"></param>
        /// <returns></returns>
        public void Copy(Variable inputVariable, NeuralNetStorage storage, NeuralNetStorage copyStorage, List<ILayerNew> layers)
        {
            // dropout layer is not used for prediction.
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="storage"></param>
        /// <param name="training"></param>
        public void Forward(NeuralNetStorage storage, bool training = true)
        {
            Dropout.Forward(Input, DropoutMask, m_random, Output, storage);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="storage"></param>
        /// <param name="random"></param>
        /// <param name="initializtion"></param>
        public void Initialize(Variable inputVariable, NeuralNetStorage storage, Random random,
            Initialization initializtion = Initialization.GlorotUniform)
        {
            m_random = new Random(random.Next());
            UpdateDimensions(inputVariable);

            DropoutMask = Variable.Create(inputVariable.Dimensions.ToArray());
            storage.AssignTensor(DropoutMask, () => DecideDropOut());
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        public void UpdateDimensions(Variable input)
        {
            Input = input;
            Output = Variable.Create(input.Dimensions.ToArray());
        }

        float DecideDropOut()
        {
            var dropOutScale = 1.0 / (1.0 - DropOut);
            return (float)(m_random.NextDouble() > DropOut ? dropOutScale : 0.0);
        }
    }
}
