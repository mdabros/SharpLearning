using System;
using CNTK;

namespace SharpLearning.Backend.Cntk
{
    internal class CntkPlaceholderOutputTensorSymbol : CntkVariableOutputTensorSymbol
    {
        public CntkPlaceholderOutputTensorSymbol( DataType dataType, ReadOnlySpan<int> shape, string name)
            : base(CNTKLib.PlaceholderVariable(shape.ToCntkShape(), dataType.ToCntk(), name),
                   dataType, shape, name)
        { }
    }
}