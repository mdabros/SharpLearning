using System;
using System.Collections.Generic;

namespace SharpLearning.Neural.Test.Backends
{
    public interface IBackend : IDisposable
    {
        Tensor Clip(Tensor norms, int v, int maxValue);

        Tensor Cast(Tensor x, DataType dataType);

        Tensor Reshape(Tensor x, int[] shape);

        Tensor Multiply(Tensor a, Tensor b, string name = null);
        Tensor Multiply<T>(T a, Tensor b, string name = null);
        Tensor Multiply<T>(Tensor a, T b, string name = null);
        Tensor Multiply(List<Tensor> batch_outs, int length);

        Tensor Divide(Tensor a, Tensor b);
        Tensor Divide<T>(T a, Tensor b);
        Tensor Divide<T>(Tensor a, T b);

        Tensor Add(Tensor a, Tensor b);
        Tensor Add<T>(Tensor a, T b);
        Tensor Add<T>(T a, Tensor b);

        Tensor Subtract(Tensor a, Tensor b, string name = null);
        Tensor Subtract<T>(Tensor a, T b, string name = null);
        Tensor Subtract<T>(T a, Tensor b, string name = null);

        Tensor Minus(Tensor tensor);

        Tensor Variable(Array array, DataType? dtype = null, string name = null);
        Tensor Variable<T>(T value, DataType? dtype = null, string name = null) where T : struct;
        Tensor Variable(Tensor tensor, DataType? dtype = null, string name = null);

        DataType? dtype(Tensor input_tensor);

        Tensor Constant<T>(T value, int[] shape = null, DataType? dtype = null, string name = null);

        Tensor Transpose(Tensor tensor);

        List<Tensor> Gradients(Tensor loss, List<Tensor> param);

        int?[] Shape(Tensor tensor);

        //Tensor bias_add(Tensor output, Tensor bias, DataFormatType? data_format = null, string name = null);

        //Tensor conv1d(Tensor inputs, Tensor kernel, int strides, PaddingType padding, DataFormatType? data_format, int dilation_rate, string name = null);

        //Tensor conv2d(Tensor inputs, Tensor kernel, int[] strides, PaddingType padding, DataFormatType? data_format, int[] dilation_rate, string name = null);

        //Tensor conv3d(Tensor inputs, Tensor kernel, int[] strides, PaddingType padding, DataFormatType? data_format, int[] dilation_rate, string name = null);
    }
}
