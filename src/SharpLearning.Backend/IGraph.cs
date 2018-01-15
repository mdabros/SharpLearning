using System;

namespace SharpLearning.Backend
{
    // Do we need IGraph *and* IBackend backend seems to have no purpose after creation...
    public interface IGraph : IDisposable
    {
        DeviceType DefaultDeviceType { get; }
    }

    //IGraph DefineGraph(int[] shape, IBackend backend)
    //{
    //    using (var graph = backend.CreateGraph())
    //    {
    //        IPlaceholderSymbol placeholder = graph.Placeholder(asdada);

    //        IParametersTensorSymbol w = graph.Parameters(shape, init);
    //        IParametersTensorSymbol b = graph.Parameters(shape, init);

    //        graph.Operators.Dense(w, b)

    //        IParametersTensorSymbol convolutionParameters = graph.Parameters();

    //        var convOperator = graph.Convolution(w, b);


    //        graph.Operators.ReLU(convOperator, shape)



    //        var pooling = graph.Operators.Pooling(convOperator, shape, "average");


    //    }

    //    using (var graph = backend.CreateLayers())
    //    {
    //        graph.Layers().
    //        layers.
    //        }
    //    return null;
    //}
}
