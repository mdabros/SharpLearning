using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLearning.Backend
{
    public static class GraphExtensions
    {
        public static ITensorSymbol Placeholder(this IGraph graph, DataType dataType, ReadOnlySpan<int> shape, string name)
        {
            // This pattern is a bit weird, but saves overloads...
            return graph.Placeholder(dataType, shape, name, graph.DefaultDeviceType);
        }
    }
}
