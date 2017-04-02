using System;
using System.Collections.Generic;
using SharpLearning.Neural.Initializations;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class InputLayer : ILayerNew
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputUnits"></param>
        public InputLayer(int inputUnits)
        {
            if (inputUnits < 1) { throw new ArgumentException("inputUnits is less than 1: " + inputUnits); }

            Input = Variable.Create(1, inputUnits); // set shape
            Output = Input;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="depth"></param>
        public InputLayer(int width, int height, int depth)
        {
            if (width < 1) { throw new ArgumentException("width is less than 1: " + width); }
            if (height < 1) { throw new ArgumentException("height is less than 1: " + height); }
            if (depth < 1) { throw new ArgumentException("depth is less than 1: " + depth); }

            Input = Variable.Create(1, depth, height, width); // set shape
            Output = Input;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        InputLayer(Variable inputVariable)
        {

            Input = inputVariable;
            Output = Input;
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
        /// <param name="executor"></param>
        public void Backward(NeuralNetStorage executor)
        {
            // do nothing, InputLayer only provides shape.
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="storage"></param>
        /// <param name="copyStorage"></param>
        /// <param name="layers"></param>
        public void Copy(Variable inputVariable, NeuralNetStorage storage, NeuralNetStorage copyStorage, List<ILayerNew> layers)
        {
            var copy = new InputLayer(inputVariable.Copy());
            copy.UpdateDimensions(inputVariable);

            layers.Add(copy);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        /// <param name="training"></param>
        public void Forward(NeuralNetStorage executor, bool training = true)
        {
            // do nothing, InputLayer only provides shape.
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="storage"></param>
        /// <param name="random"></param>
        /// <param name="initializtion"></param>
        public void Initialize(Variable inputVariable, NeuralNetStorage storage, Random random, Initialization initializtion)
        {
            UpdateDimensions(inputVariable);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        public void UpdateDimensions(Variable input)
        {
            Input = input;
            Output = Input;
        }
    }
}
