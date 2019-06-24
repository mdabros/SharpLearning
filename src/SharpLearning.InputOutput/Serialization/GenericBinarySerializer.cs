using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace SharpLearning.InputOutput.Serialization
{

    /// <summary>
    /// Generic xml serializer using BinaryFormatter
    /// </summary>
    public sealed class GenericBinarySerializer : IGenericSerializer
    {
        /// <summary>
        /// Deserialize data from the provided reader
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="reader"></param>
        /// <returns></returns>
        public T Deserialize<T>(Func<TextReader> reader)
        {
            var serializer = new BinaryFormatter();

            using (var baseReader = reader())
            {
                if (baseReader is StreamReader)
                {
                    var baseStream = ((StreamReader)baseReader).BaseStream;
                    return (T)serializer.Deserialize(baseStream);
                }
                else if (baseReader is StringReader baseStream)
                {
                    var bytes = Convert.FromBase64String(baseStream.ReadToEnd());
                    using (var memoryStream = new MemoryStream(bytes))
                    {
                        return (T)serializer.Deserialize(memoryStream);
                    }
                }
                else
                {
                    throw new ArgumentException("Unsupported reader type: " + baseReader.GetType());
                }
            }
        }

        /// <summary>
        /// Serialize data to the provided writer
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="data"></param>
        /// <param name="writer"></param>
        public void Serialize<T>(T data, Func<TextWriter> writer)
        {
            var serializer = new BinaryFormatter();

            using (var baseWriter = writer())
            {
                if(baseWriter is StreamWriter)
                {
                    var baseStream = ((StreamWriter)baseWriter).BaseStream;
                    serializer.Serialize(baseStream, data);
                }
                else if(baseWriter is StringWriter)
                {
                    using (var memoryStream = new MemoryStream())
                    {
                        serializer.Serialize(memoryStream, data);
                        baseWriter.Write(Convert.ToBase64String(memoryStream.ToArray()));
                    }
                }
                else
                {
                    throw new ArgumentException("Unsupported writer type: " + baseWriter.GetType());
                }
            }
        }
    }
}
