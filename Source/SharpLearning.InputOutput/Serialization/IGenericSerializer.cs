using System;
using System.IO;

namespace SharpLearning.InputOutput.Serialization
{
    /// <summary>
    /// Generic serializer interface
    /// </summary>
    public interface IGenericSerializer
    {
        /// <summary>
        /// Serialize data to the provided writer
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="data"></param>
        /// <param name="writer"></param>
        void Serialize<T>(T data, Func<TextWriter> writer);

        /// <summary>
        /// Deserialize data from the provided reader
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="reader"></param>
        /// <returns></returns>
        T Deserialize<T>(Func<TextReader> reader);
    }
}
