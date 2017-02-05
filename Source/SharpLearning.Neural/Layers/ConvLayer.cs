using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;
using System;
using System.Collections.Generic;

namespace SharpLearning.Neural.Layers
{
    /// <summary>
    /// Convolutional layer using GEMM implementation 
    /// based on: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
    /// and: https://arxiv.org/pdf/1410.0759.pdf
    /// </summary>
    [Serializable]
    public sealed class ConvLayer : ILayer, IBatchNormalizable
    {
        /// <summary>
        /// 
        /// </summary>
        public int Width { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public int Height { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public int Depth { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public Activation ActivationFunc { get; set; }

        /// <summary>
        /// Does the layer use batch normalization
        /// </summary>
        public bool UseBatchNormalization { get; set; }

        int m_padding = 0;
        int m_stride = 1;

        /// <summary>
        /// 
        /// </summary>
        public readonly int FilterWidth;

        /// <summary>
        /// 
        /// </summary>
        public readonly int FilterHeight;

        /// <summary>
        /// 
        /// </summary>
        public readonly int FilterCount;

        /// <summary>
        /// Weights in the layer.
        /// </summary>
        public Matrix<float> Weights;

        /// <summary>
        /// Biases in the layer.
        /// </summary>
        public Vector<float> Bias;

        /// <summary>
        /// Weight gradients.
        /// </summary>
        public Matrix<float> WeightsGradients;

        /// <summary>
        /// Bias gradients.
        /// </summary>
        public Vector<float> BiasGradients;

        /// <summary>
        /// 
        /// </summary>
        public Matrix<float> OutputActivations;

        Matrix<float> m_inputActivations;
        Matrix<float> m_delta;

        /// <summary>
        /// 
        /// </summary>
        public int InputHeight;

        /// <summary>
        /// 
        /// </summary>
        public int InputWidth;

        /// <summary>
        /// 
        /// </summary>
        public int InputDepth;

        /// <summary>
        /// Member for storing the image to columns conversion of the minibatch.
        /// </summary>
        public Matrix<float> Im2Cols;

        /// <summary>
        /// Member for storing  the convolution.
        /// </summary>
        public Matrix<float> Conv;
        Matrix<float> m_deltaInReshape;

        /// <summary>
        /// Convolutional layer using GEMM implementation 
        /// based on: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
        /// and: https://arxiv.org/pdf/1410.0759.pdf
        /// </summary>
        /// <param name="filterWidth">The width of the filters</param>
        /// <param name="filterHeight">The height of the filters</param>
        /// <param name="filterCount">The number of filters</param>
        /// <param name="stride">Controls the distance between each neighbouring filter (default is 1).</param>
        /// <param name="padding">Controls the padding at the edges of the input (default is 0).</param>
        /// <param name="activation">Type of activation function used (default is Relu).</param>
        public ConvLayer(int filterWidth, int filterHeight, int filterCount, int stride = 1, int padding = 0,
            Activation activation = Activation.Relu)
        {
            FilterWidth = filterWidth;
            FilterHeight = filterHeight;
            FilterCount = filterCount;

            ActivationFunc = activation;
            m_stride = stride;
            m_padding = padding;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="delta"></param>
        /// <returns></returns>
        public Matrix<float> Backward(Matrix<float> delta)
        {
            // Reshape delta to fit with data layout in im2col
            ConvUtils.ReshapeConvolutionsToRowMajor(delta, InputDepth, InputHeight, InputWidth,
                FilterWidth, FilterHeight, m_padding, m_padding, m_stride, m_stride, m_deltaInReshape);

            // Calculate gradients for weights and biases
            m_deltaInReshape.TransposeAndMultiply(Im2Cols, WeightsGradients);
            m_deltaInReshape.SumRows(BiasGradients);

            // calcualte delta for next layer.
            Weights.TransposeThisAndMultiply(m_deltaInReshape, Im2Cols);

            // convert back to original layout
            m_delta.Clear();
            ConvUtils.Batch_Col2Im(Im2Cols, InputDepth, InputHeight, InputWidth,
                FilterHeight, FilterWidth, m_padding, m_padding, m_stride, m_stride, m_delta);

            return m_delta;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix<float> Forward(Matrix<float> input)
        {
            m_inputActivations = input;

            // Arrange input item for GEMM version of convolution.
            ConvUtils.Batch_Im2Col(m_inputActivations, InputDepth, InputHeight, InputWidth,
                FilterWidth, FilterHeight, m_padding, m_padding, m_stride, m_stride, Im2Cols);

            // matrix multiplication for convolution
            Weights.Multiply(Im2Cols, Conv);
            Conv.AddColumnWise(Bias, Conv);

            // Return the covolved data to row major and copy  data to output
            ConvUtils.ReshapeConvolutionsToRowMajor(Conv, InputDepth, InputHeight, InputWidth,
                FilterWidth, FilterHeight, m_padding, m_padding, m_stride, m_stride, OutputActivations);

            return OutputActivations;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="inputDepth"></param>
        /// <param name="batchSize"></param>
        /// <param name="initializtion"></param>
        /// <param name="random"></param>

        public void Initialize(int inputWidth, int inputHeight, int inputDepth, int batchSize, Initialization initializtion, Random random)
        {
            InputHeight = inputHeight;
            InputWidth = inputWidth;
            InputDepth = inputDepth;           

            var filterGridWidth = ConvUtils.GetFilterGridLength(InputWidth, FilterWidth, m_stride, m_padding);
            var filterGridHeight = ConvUtils.GetFilterGridLength(InputHeight, FilterHeight, m_stride, m_padding);

            // Calculations of dimensions based on:
            // Nvidia, cuDNN: Efficient Primitives for Deep Learning: https://arxiv.org/pdf/1410.0759.pdf
            var filterCubeSize = InputDepth * FilterWidth * FilterHeight;
            var filterGridSize = filterGridWidth * filterGridHeight;

            Height = filterGridHeight;
            Width = filterGridWidth;
            Depth = FilterCount;

            var fans = WeightInitialization.GetFans(this, InputWidth, InputHeight, inputDepth);
            var distribution = WeightInitialization.GetWeightDistribution(initializtion, fans, random);

            Weights = Matrix<float>.Build.Random(FilterCount, filterCubeSize, distribution);
            WeightsGradients = Matrix<float>.Build.Dense(FilterCount, filterCubeSize);

            Bias = Vector<float>.Build.Dense(FilterCount, 0.0f);
            BiasGradients = Vector<float>.Build.Dense(FilterCount);

            Im2Cols = Matrix<float>.Build.Dense(filterCubeSize, filterGridSize * batchSize);
            Conv = Matrix<float>.Build.Dense(FilterCount, filterGridSize * batchSize);
            
            OutputActivations = Matrix<float>.Build.Dense(batchSize, FilterCount * filterGridSize);
            m_deltaInReshape = Matrix<float>.Build.Dense(FilterCount, filterGridSize * batchSize);

            var fanIn = inputWidth * inputHeight * inputDepth;
            m_delta = Matrix<float>.Build.Dense(batchSize, fanIn);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parametersAndGradients"></param>
        public void AddParameresAndGradients(List<ParametersAndGradients> parametersAndGradients)
        {
            var all = new ParametersAndGradients(GetParameters(), GetGradients());
            parametersAndGradients.Add(all);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public WeightsAndBiases GetGradients()
        {
            return new WeightsAndBiases(WeightsGradients, BiasGradients);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public WeightsAndBiases GetParameters()
        {
            return new WeightsAndBiases(Weights, Bias);
        }

        /// <summary>
        /// Copies a minimal version of the layer to be used in a model for predictions.
        /// </summary>
        /// <param name="layers"></param>
        public void CopyLayerForPredictionModel(List<ILayer> layers)
        {
            var batchSize = 1; // prediction time only uses 1 item at a time.
            var copy = new ConvLayer(FilterWidth, FilterHeight, FilterCount, m_stride, m_padding, ActivationFunc);

            copy.InputDepth = InputDepth;
            copy.InputWidth = InputWidth;
            copy.InputHeight = InputHeight;

            var filterGridWidth = ConvUtils.GetFilterGridLength(InputWidth, FilterWidth, m_stride, m_padding);
            var filterGridHeight = ConvUtils.GetFilterGridLength(InputHeight, FilterHeight, m_stride, m_padding);

            // Calculations of dimensions based on:
            // Nvidia, cuDNN: Efficient Primitives for Deep Learning: https://arxiv.org/pdf/1410.0759.pdf
            var filterCubeSize = InputDepth * FilterWidth * FilterHeight;
            var filterGridSize = filterGridWidth * filterGridHeight;

            copy.Width = this.Width;
            copy.Height = this.Height;
            copy.Depth = this.Depth;

            var fanOut = Width * Height * Depth;
            
            copy.Weights = Matrix<float>.Build.Dense(Weights.RowCount, Weights.ColumnCount);
            copy.Bias = Vector<float>.Build.Dense(Bias.Count);
            Array.Copy(Weights.Data(), copy.Weights.Data(), Weights.Data().Length);
            Array.Copy(Bias.Data(), copy.Bias.Data(), Bias.Data().Length);

            copy.Im2Cols = Matrix<float>.Build.Dense(filterCubeSize, filterGridSize * batchSize);
            copy.Conv = Matrix<float>.Build.Dense(FilterCount, filterGridSize * batchSize);
            copy.OutputActivations = Matrix<float>.Build.Dense(batchSize, fanOut);

            layers.Add(copy);
        }
    }
}
