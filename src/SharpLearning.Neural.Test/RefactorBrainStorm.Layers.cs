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
        void Initialize(Variable inputVariable, int batchSize,
            NeuralNetStorage storage, Random random, 
            Initialization initializtion);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        void UpdateDimensions(int batchSize);

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
        public void Initialize(Variable inputVariable, int batchSize,
            NeuralNetStorage storage, Random random, 
            Initialization initializtion)
        {
            UpdateDimensions(batchSize);
        }

        public int ParameterCount()
        {
            return 0;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        public void UpdateDimensions(int batchSize)
        {
            var dimensions = new int[] { batchSize }.Concat(Input.Dimensions).ToArray();
            Input = Variable.Create(dimensions);
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
        public void Initialize(Variable inputVariable, int batchSize,
            NeuralNetStorage storage, Random random,
            Initialization initializtion = Initialization.GlorotUniform)
        {
            Input = inputVariable;
            UpdateDimensions(batchSize);

            // Assumes first dimension is batch size
            var fanIn = inputVariable.DimensionOffSets[0];
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
        public void UpdateDimensions(int batchSize)
        {
            Output = Variable.Create(batchSize, m_units);
        }

        public int ParameterCount()
        {
            var weightCount = Weights.ElementCount;
            var biasCount = Bias.ElementCount;

            return weightCount + biasCount;
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public sealed class ConvolutionLayer : ILayer
    {
        readonly Operators.ConvolutionDescriptor m_convolutionDescriptor;

        Variable Weights;
        Variable Bias;

        Variable Im2Col;
        Variable Conv;

        BorderMode m_borderMode;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="filterChannels"></param>
        /// <param name="filterH"></param>
        /// <param name="filterW"></param>
        /// <param name="strideH"></param>
        /// <param name="strideW"></param>
        /// <param name="padH"></param>
        /// <param name="padW"></param>
        public ConvolutionLayer(int filterChannels, int filterH, int filterW,
                int strideH, int strideW,
                int padH, int padW, BorderMode borderMode)
        {
            m_convolutionDescriptor = new Operators.ConvolutionDescriptor(filterChannels, 
                filterH, filterW, 
                strideH, strideW, 
                padH, padW);

            m_borderMode = borderMode;
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
            Operators.Convolution.Forward(Input, Im2Col, Conv, m_convolutionDescriptor,
                Weights, Bias, m_borderMode, Output, storage);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="storage"></param>
        public void Backward(NeuralNetStorage storage)
        {
            Operators.Convolution.Backward(Input, Im2Col, Conv, m_convolutionDescriptor, 
                Weights, Bias, m_borderMode, Output, storage);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputVariable"></param>
        /// <param name="storage"></param>
        /// <param name="random"></param>
        /// <param name="initializtion"></param>
        public void Initialize(Variable inputVariable, int batchSize,
            NeuralNetStorage storage, Random random,
            Initialization initializtion = Initialization.GlorotUniform)
        {
            UpdateDimensions(batchSize);

            var c = inputVariable.Dimensions[1];
            var h = inputVariable.Dimensions[2];
            var w = inputVariable.Dimensions[3];

            // Calculations of dimensions based on:
            // Nvidia, cuDNN: Efficient Primitives for Deep Learning: https://arxiv.org/pdf/1410.0759.pdf
            var filterCubeSize = c * m_convolutionDescriptor.FilterW * m_convolutionDescriptor.FilterH;
            var receptiveFieldSize = m_convolutionDescriptor.FilterW * m_convolutionDescriptor.FilterH;

            var fanIn = filterCubeSize;
            var fanOut = m_convolutionDescriptor.FilterChannels * receptiveFieldSize;
            var fans = new FanInFanOut(fanIn, fanOut);
            var distribution = WeightInitialization.GetWeightDistribution(initializtion, fans, random);

            Weights = Variable.CreateTrainable(m_convolutionDescriptor.FilterChannels, filterCubeSize);
            storage.AssignTensor(Weights, () => (float)distribution.Sample());

            Bias = Variable.CreateTrainable(m_convolutionDescriptor.FilterChannels);
            storage.AssignTensor(Bias, () => 0.0f);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        public void UpdateDimensions(int batchSize)
        {
            //Input = input;

            //var batchSize = Input.Dimensions[0];
            var c = Input.Dimensions[1];
            var h = Input.Dimensions[2];
            var w = Input.Dimensions[3];

            var filterGridWidth = ConvUtils.GetFilterGridLength(w, m_convolutionDescriptor.FilterW,
                m_convolutionDescriptor.StrideW, m_convolutionDescriptor.PadW, m_borderMode);
            var filterGridHeight = ConvUtils.GetFilterGridLength(h, m_convolutionDescriptor.FilterH,
                m_convolutionDescriptor.StrideH, m_convolutionDescriptor.PadW, m_borderMode);

            // Calculations of dimensions based on:
            // Nvidia, cuDNN: Efficient Primitives for Deep Learning: https://arxiv.org/pdf/1410.0759.pdf
            var filterCubeSize = c * m_convolutionDescriptor.FilterH * m_convolutionDescriptor.FilterW;
            var filterGridSize = filterGridWidth * filterGridHeight;

            Im2Col = Variable.Create(filterGridSize * batchSize, filterCubeSize);

            Conv = Variable.Create(m_convolutionDescriptor.FilterChannels, batchSize,
                filterGridHeight, filterGridWidth);

            Output = Variable.Create(batchSize, m_convolutionDescriptor.FilterChannels,
                filterGridHeight, filterGridWidth);
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
        public void Initialize(Variable inputVariable, int batchSize,
            NeuralNetStorage storage, Random random,
            Initialization initializtion)
        {
            Input = inputVariable;
            UpdateDimensions(batchSize);
        }

        /// <summary>
        /// Updates the input and output dimensions of the layer
        /// </summary>
        /// <param name="input"></param>
        public void UpdateDimensions(int batchSize)
        {
            Output = Variable.Create(Input.Dimensions.ToArray());
        }

        public int ParameterCount()
        {
            return 0;
        }
    }
}
