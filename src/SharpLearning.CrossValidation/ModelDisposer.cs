using System;
using SharpLearning.Common.Interfaces;

namespace SharpLearning.CrossValidation
{
    internal static class ModelDisposer
    {
        internal static void DisposeIfDisposable<TPrediction>(IPredictorModel<TPrediction> model)
        {
            var modelDisposable = typeof(IDisposable).IsAssignableFrom(model.GetType());
            if (modelDisposable)
            {
                ((IDisposable)model).Dispose();
            }
        }
    }
}
