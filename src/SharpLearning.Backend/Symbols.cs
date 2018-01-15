using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLearning.Backend
{
    public interface ISymbol : IDisposable { }

    public interface ITensorSymbol : ISymbol
    {

    }

    public interface IOutputTensorSymbol : ITensorSymbol { }

    public interface IParameterTensorSymbol : ITensorSymbol { }

    public interface IOperatorSymbol { }

}
