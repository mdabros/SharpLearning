using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;

namespace SharpLearning.Optimization.Test.StagedOptimization
{
    public class Repository : IRepository
    {
        const string EmptyName = "";
        readonly Dictionary<InstanceKey, object> m_keyToValue = new Dictionary<InstanceKey, object>();

        public void Add<T>(T value) => Add(EmptyName, value);

        public void Add<T>(string name, T value)
        {
            var type = typeof(T);
            var key = InstanceKey.New(type, name);

            if (m_keyToValue.ContainsKey(key))
            {
                throw new ArgumentException($"Type: {type} with key: {key} already exist on repository");
            }

            m_keyToValue.Add(key, value);
        }

        public bool TryGet<T>(out T value) => TryGet(EmptyName, out value);

        public bool TryGet<T>(string name, out T value)
        {
            var type = typeof(T);
            var key = InstanceKey.New(type, name);

            object o;
            if (m_keyToValue.TryGetValue(key, out o))
            {
                value = (T)o;
                return true;
            }
            value = default(T);
            return false;
        }

        public T Get<T>() => Get<T>(EmptyName);

        public T Get<T>(string name)
        {
            T value;
            if (TryGet<T>(name, out value))
            {
                return value;
            }
            throw new KeyNotFoundException($"Could not find type: {typeof(T).FullName} with name: {name}");
        }

        public void Replace<T>(T value) => Replace(EmptyName, value);

        public void Replace<T>(string name, T value)
        {
            var type = typeof(T);
            var key = InstanceKey.New(type, name);

            m_keyToValue[key] = value;
        }

        public IEnumerable<(object obj, Type type, string name)> GetAll() =>
            m_keyToValue.Select(kvp => (kvp.Value, kvp.Key.Type, kvp.Key.Name));

        public IEnumerable<object> GetAllValues() => m_keyToValue.Values;

        [DebuggerDisplay("Type = {Type} Name = {Name}")]
        struct InstanceKey : IEquatable<InstanceKey>
        {
            public readonly Type Type;
            public readonly string Name;

            public InstanceKey(Type type, string name)
            {
                Type = type ?? throw new ArgumentNullException(nameof(type));
                Name = name ?? throw new ArgumentNullException(nameof(name));
            }

            public static InstanceKey New(Type type, string name)
            {
                return new InstanceKey(type, name);
            }

            public bool Equals(InstanceKey other)
            {
                return Type.Equals(other.Type) && Name.Equals(other.Name);
            }

            public override bool Equals(object other)
            {
                if (other == null)
                { return false; }

                if (other.GetType() != typeof(InstanceKey))
                { return false; }

                return Equals((InstanceKey)other);
            }

            public override int GetHashCode()
            {
                return Type.GetHashCode() ^ Name.GetHashCode();
            }

            public override string ToString()
            {
                return string.Format("Type = {0} Name = '{1}'", Type, Name);
            }
        }
    }
}
