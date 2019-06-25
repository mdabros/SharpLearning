using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Initializations;

namespace SharpLearning.Neural.Layers
{
    /// <summary>
    /// 2D Convolutional layer using GEMM implementation 
    /// based on: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
    /// and: https://arxiv.org/pdf/1410.0759.pdf
    /// </summary>
    [Serializable]
    public sealed class Conv2DLayer : ILayer, IBatchNormalizable
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
        public bool BatchNormalization { get; set; }

        readonly int m_padWidth = 0;
        readonly int m_padHeight = 0;
        readonly int m_stride = 1;

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
        /// Border mode used for convolution.
        /// </summary>
        public BorderMode BorderMode;

        /// <summary>
        /// 2D Convolutional layer using GEMM implementation 
        /// based on: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
        /// and: https://arxiv.org/pdf/1410.0759.pdf
        /// </summary>
        /// <param name="filterWidth">The width of the filters</param>
        /// <param name="filterHeight">The height of the filters</param>
        /// <param name="filterCount">The number of filters</param>
        /// <param name="stride">Controls the distance between each neighboring filter (default is 1)</param>
        /// <param name="padWidth">Zero padding for the width dimension (default is 0)</param>
        /// <param name="padHeight">Zero padding for the height dimension (default is 0)</param>
        /// <param name="activation">Type of activation function used (default is Relu)</param>
        public Conv2DLayer(int filterWidth, int filterHeight, int filterCount, int stride, 
            int padWidth, int padHeight, Activation activation = Activation.Relu)
        {
            if (filterWidth < 1) { throw new ArgumentException("filterWidth is less than 1: " + filterWidth); }
            if (filterHeight < 1) { throw new ArgumentException("poolHeight is less than 1: " + filterHeight); }
            if (filterCount < 1) { throw new ArgumentException("filterCount is less than 1: " + filterCount); }
            if (padWidth < 0) { throw new ArgumentException("padWidth is less than 0: " + padWidth); }
            if (padHeight < 0) { throw new ArgumentException("padHeight is less than 0: " + padHeight); }
            if (stride < 1) { throw new ArgumentException("stride is less than 0: " + stride); }


            FilterWidth = filterWidth;
            FilterHeight = filterHeight;
            FilterCount = filterCount;

            ActivationFunc = activation;
            m_stride = stride;
            m_padWidth = padWidth;
            m_padHeight = padHeight;
            BorderMode = BorderMode.Undefined;
        }

        /// <summary>
        /// 2D Convolutional layer using GEMM implementation 
        /// based on: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
        /// and: https://arxiv.org/pdf/1410.0759.pdf
        /// </summary>
        /// <param name="filterWidth">The width of the filters</param>
        /// <param name="filterHeight">The height of the filters</param>
        /// <param name="filterCount">The number of filters</param>
        /// <param name="stride">Controls the distance between each neighboring filter (default is 1)</param>
        /// <param name="borderMode">Border mode of the convolutional operation. 
        /// This will set the width and height padding automatically based on the selected border mode: Valid, Same or Full (default is Valid)</param>
        /// <param name="activation">Type of activation function used (default is Relu)</param>
        public Conv2DLayer(int filterWidth, int filterHeight, int filterCount, int stride = 1, 
            BorderMode borderMode = BorderMode.Valid, Activation activation = Activation.Relu)
            : this(filterWidth, filterHeight, filterCount, stride,
                  ConvUtils.PaddingFromBorderMode(filterWidth, borderMode),
                  ConvUtils.PaddingFromBorderMode(filterHeight, borderMode))
        {
            BorderMode = borderMode;
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
                FilterWidth, FilterHeight, m_padHeight, m_padWidth,
                m_stride, m_stride, BorderMode, m_deltaInReshape);

            // Calculate gradients for weights and biases
            m_deltaInReshape.TransposeAndMultiply(Im2Cols, WeightsGradients);
            m_deltaInReshape.SumRows(BiasGradients);

            // calcualte delta for next layer.
            Weights.TransposeThisAndMultiply(m_deltaInReshape, Im2Cols);

            // convert back to original layout
            m_delta.Clear();
            ConvUtils.Batch_Col2Im(Im2Cols, InputDepth, InputHeight, InputWidth,
                FilterHeight, FilterWidth, m_padHeight, m_padWidth, 
                m_stride, m_stride, BorderMode, m_delta);

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
                FilterWidth, FilterHeight, m_padHeight, m_padWidth, 
                m_stride, m_stride, BorderMode, Im2Cols);

            // matrix multiplication for convolution
            Weights.Multiply(Im2Cols, Conv);
            Conv.AddColumnWise(Bias, Conv);

            // Return the convolved data to row major and copy  data to output
            ConvUtils.ReshapeConvolutionsToRowMajor(Conv, InputDepth, InputHeight, InputWidth,
                FilterWidth, FilterHeight, m_padHeight, m_padWidth, 
                m_stride, m_stride, BorderMode, OutputActivations);

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

        public void Initialize(int inputWidth, int inputHeight, int inputDepth, int batchSize, 
            Initialization initializtion, Random random)
        {
            InputHeight = inputHeight;
            InputWidth = inputWidth;
            InputDepth = inputDepth;           

            var filterGridWidth = ConvUtils.GetFilterGridLength(InputWidth, FilterWidth, 
                m_stride, m_padWidth, BorderMode);
            var filterGridHeight = ConvUtils.GetFilterGridLength(InputHeight, FilterHeight, 
                m_stride, m_padHeight, BorderMode);

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
            var weights = new ParametersAndGradients(Weights.Data(), WeightsGradients.Data());
            var bias = new ParametersAndGradients(Bias.Data(), BiasGradients.Data());
            parametersAndGradients.Add(weights);
            parametersAndGradients.Add(bias);
        }

        /// <summary>
        /// Copies a minimal version of the layer to be used in a model for predictions.
        /// </summary>
        /// <param name="layers"></param>
        public void CopyLayerForPredictionModel(List<ILayer> layers)
        {
            var batchSize = 1; // prediction time only uses 1 item at a time.
            var copy = new Conv2DLayer(FilterWidth, FilterHeight, FilterCount, 
                m_stride, m_padWidth, m_padHeight, ActivationFunc);

            copy.InputDepth = InputDepth;
            copy.InputWidth = InputWidth;
            copy.InputHeight = InputHeight;

            var filterGridWidth = ConvUtils.GetFilterGridLength(InputWidth, FilterWidth, 
                m_stride, m_padWidth, BorderMode);
            var filterGridHeight = ConvUtils.GetFilterGridLength(InputHeight, FilterHeight, 
                m_stride, m_padHeight, BorderMode);
            copy.BorderMode = BorderMode;

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
