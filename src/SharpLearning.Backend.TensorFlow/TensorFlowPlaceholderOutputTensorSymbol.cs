using System;
using TensorFlow;

namespace SharpLearning.Backend.TensorFlow
{
    internal class TensorFlowPlaceholderOutputTensorSymbol : TensorFlowOutputTensorSymbol
    {
        public TensorFlowPlaceholderOutputTensorSymbol(TFGraph graph, DataType dataType, ReadOnlySpan<int> shape, string name)
            : base(graph.Placeholder(dataType.ToTF(), shape.ToTFShape(), name),
                   dataType, shape, name)
        { }
    }
}