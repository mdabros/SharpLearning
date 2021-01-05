using System;
using System.Collections.Generic;

namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public interface IRepository
    {
        T Get<T>();
        T Get<T>(string name);

        bool TryGet<T>(out T value);
        bool TryGet<T>(string name, out T value);

        void Add<T>(T value);
        void Add<T>(string name, T value);

        void Replace<T>(T value);
        void Replace<T>(string name, T value);

        IEnumerable<(object obj, Type type, string name)> GetAll();
        IEnumerable<object> GetAllValues();
    }
}
