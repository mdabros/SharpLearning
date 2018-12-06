using System;
using System.Linq;
using System.Collections.Generic;

namespace SharpLearning.Neural.Test.RefactorBranStorm
{
    /// <summary>
    /// 
    /// </summary>
    public interface ILayer
    {
        /// <summary>
        /// Layer output shape
        /// </summary>
        Variable Output { get; set; }

        /// <summary>
        /// Layer Input shape
        /// </summary>
        Variable Input { get; set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="storage"></param>
        /// <param name="training"></param>
        void Forward(NeuralNetStorage storage);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="storage"></param>
        void Backward(NeuralNetStorage storage);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="storage"></param>
        /// <param name="random"></param>
        /// <param name="initializtion"></param>
        void Initialize(Variable inputVariable, NeuralNetStorage storage, Random random, Initialization initializtion);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        void UpdateDimensions(Variable input);

        int ParameterCount();
    }

    /// <summary>
    /// 
    /// </summary>
    public sealed class InputLayer : ILayer
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="depth"></param>
        public InputLayer(params int[] inputDimensions)
        {
            if (inputDimensions == null) throw new ArgumentNullException(nameof(inputDimensions));

            Input = Variable.Create(inputDimensions); // set shape
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
        /// <param name="storage"></param>
        public void Backward(NeuralNetStorage storage)
        {
            // do nothing, InputLayer only provides shape.
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="storage"></param>
        /// <param name="training"></param>
        public void Forward(NeuralNetStorage storage)
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

        public int ParameterCount()
        {
            return 0;
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

    /// <summary>
    /// 
    /// </summary>
    public sealed class DenseLayer : ILayer
    {
        readonly int m_units;

        Variable Weights;
        Variable Bias;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="units"></param>
        public DenseLayer(int units)
        {
            if (units < 1) { throw new ArgumentException("HiddenLayer must have at least 1 hidden unit"); }
            m_units = units;
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
        /// <param name="training"></param>
        public void Forward(NeuralNetStorage storage)
        {
            Operators.Dense.Forward(Input, Weights, Bias,
                Output, storage);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="storage"></param>
        public void Backward(NeuralNetStorage storage)
        {
            Operators.Dense.Backward(Input, Weights, Bias,
                Output, storage);
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
            UpdateDimensions(inputVariable);

            // Assumes no batch dimension in inputVariable.
            var fanIn = inputVariable.Dimensions.Aggregate((v1, v2) => v1 * v2);
            var fanOut = m_units;

            var fans = new FanInFanOut(fanIn, fanOut);
            var distribution = WeightInitialization.GetWeightDistribution(initializtion, fans, random);

            Weights = Variable.CreateTrainable(fans.FanIn, fans.FanOut);
            storage.AssignTensor(Weights, () => (float)distribution.Sample());

            Bias = Variable.CreateTrainable(fans.FanOut);
            storage.AssignTensor(Bias, () => 0.0f);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        public void UpdateDimensions(Variable input)
        {
            Input = input;

            var batchSize = input.Dimensions[0];
            Output = Variable.Create(batchSize, m_units);
        }

        public int ParameterCount()
        {
            var weightCount = Weights.ElementCount;
            var biasCount = Bias.ElementCount;

            return weightCount + biasCount;
        }
    }

    public sealed class ReluLayer : ActivationLayer
    {
        public ReluLayer()
        {
            m_forward = Operators.ReLU.Forward;
            m_backward = Operators.ReLU.Backward;
        }
    }

    public sealed class SigmoidLayer : ActivationLayer
    {
        public SigmoidLayer()
        {
            m_forward = Operators.Sigmoid.Forward;
            m_backward = Operators.Sigmoid.Backward;
        }
    }

    /// <summary>
    /// Activation layer. Adds activation functions to a neural net.
    /// </summary>
    public abstract class ActivationLayer : ILayer
    {
        protected Action<Variable, Variable, NeuralNetStorage> m_forward;
        protected Action<Variable, Variable, NeuralNetStorage> m_backward;

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
        /// <param name="training"></param>
        public void Forward(NeuralNetStorage storage)
        {
            m_forward(Input, Output, storage);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="storage"></param>
        public void Backward(NeuralNetStorage storage)
        {
            m_backward(Input, Output, storage);
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
            UpdateDimensions(inputVariable);
        }

        /// <summary>
        /// Updates the input and output dimensions of the layer
        /// </summary>
        /// <param name="input"></param>
        public void UpdateDimensions(Variable input)
        {
            Input = input;
            Output = Variable.Create(input.Dimensions.ToArray());
        }

        public int ParameterCount()
        {
            return 0;
        }
    }
}
