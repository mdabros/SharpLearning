using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using System.Xml;

namespace SharpLearning.InputOutput.Serialization;

/// <summary>
/// Generic xml serializer using DataContractSerializer
/// </summary>
public sealed class GenericXmlDataContractSerializer : IGenericSerializer
{
    readonly Type[] m_knownTypes;
    readonly bool m_preserveObjectReferences;

    /// <summary>
    /// Generic xml serializer using DataContractSerializer
    /// </summary>
    /// <param name="knownTypes">If the serializer fails with an unknown type exception. 
    /// The necesarry types can be provided in the cosntructer.</param>
    /// <param name="preserveObjectReferences">This parameter controls if object references should be preserved in the serialization (default is true). 
    /// This adds extra information to the xml which is needed when serializing some model types. 
    /// Currently only the SharpLearning.Neural models require this.</param>
    public GenericXmlDataContractSerializer(Type[] knownTypes, bool preserveObjectReferences = true)
    {
        m_knownTypes = knownTypes ?? throw new ArgumentNullException(nameof(knownTypes));
        m_preserveObjectReferences = preserveObjectReferences;
    }

    /// <summary>
    /// Generic xml serializer using DataContractSerializer
    /// </summary>
    /// <param name="preserveObjectReferences">This parameter controls if object references should be preserved in the serialization (default is true). 
    /// This adds extra information to the xml which is needed when serializing some model types. 
    /// Currently only the SharpLearning.Neural models require this.</param>
    public GenericXmlDataContractSerializer(bool preserveObjectReferences = true)
       : this([], preserveObjectReferences)
    {
    }

    /// <summary>
    /// Generic xml serializer using DataContractSerializer
    /// </summary>
    public GenericXmlDataContractSerializer()
       : this([], true)
    {
    }

    /// <summary>
    /// Serialize data to the provided writer
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="data"></param>
    /// <param name="writer"></param>
    public void Serialize<T>(T data, Func<TextWriter> writer)
    {
        var settings = new XmlWriterSettings { Indent = true };

        using var texWriter = writer();
        using var xmlWriter = XmlWriter.Create(texWriter, settings);
        var serializer = new DataContractSerializer(typeof(T), new DataContractSerializerSettings()
        {
            KnownTypes = m_knownTypes,
            MaxItemsInObjectGraph = int.MaxValue,
            IgnoreExtensionDataObject = false,
            PreserveObjectReferences = m_preserveObjectReferences,
            DataContractResolver = new GenericResolver()
        });

        serializer.WriteObject(xmlWriter, data);
    }

    /// <summary>
    /// Deserialize data from the provided reader
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="reader"></param>
    /// <returns></returns>
    public T Deserialize<T>(Func<TextReader> reader)
    {
        using var textReader = reader();
        using var xmlReader = XmlReader.Create(textReader);
        var serializer = new DataContractSerializer(typeof(T), new DataContractSerializerSettings()
        {
            KnownTypes = m_knownTypes,
            MaxItemsInObjectGraph = int.MaxValue,
            IgnoreExtensionDataObject = false,
            PreserveObjectReferences = m_preserveObjectReferences,
            DataContractResolver = new GenericResolver()
        });

        return (T)serializer.ReadObject(xmlReader);
    }

    #region GenericResolver

    /// <summary>
    /// Internal class for generic object type resolving
    /// </summary>
    internal class GenericResolver : DataContractResolver
    {
        const string DefaultNamespace = "global";

        readonly Dictionary<Type, Tuple<string, string>> m_typeToNames;
        readonly Dictionary<string, Dictionary<string, Type>> m_namesToType;

        public Type[] KnownTypes => m_typeToNames.Keys.ToArray();

        public GenericResolver()
            : this(ReflectTypes())
        { }

        public GenericResolver(Type[] typesToResolve)
        {
            m_typeToNames = [];
            m_namesToType = [];

            foreach (Type type in typesToResolve)
            {
                var typeNamespace = GetNamespace(type);
                var typeName = GetName(type);

                m_typeToNames[type] = new Tuple<string, string>(typeNamespace, typeName);

                if (m_namesToType.ContainsKey(typeNamespace) == false)
                {
                    m_namesToType[typeNamespace] = [];
                }
                m_namesToType[typeNamespace][typeName] = type;
            }
        }

        public static GenericResolver Merge(GenericResolver resolver1, GenericResolver resolver2)
        {
            if (resolver1 == null)
            {
                return resolver2;
            }
            if (resolver2 == null)
            {
                return resolver1;
            }
            var types = new List<Type>();
            types.AddRange(resolver1.KnownTypes);
            types.AddRange(resolver2.KnownTypes);

            return new GenericResolver(types.ToArray());
        }

        public override Type ResolveName(string typeName,
            string typeNamespace,
            Type declaredType,
            DataContractResolver knownTypeResolver)
        {
            if (m_namesToType.TryGetValue(typeNamespace, out var nameToType))
            {
                if (nameToType.TryGetValue(typeName, out var value))
                {
                    return value;
                }
            }
            return knownTypeResolver.ResolveName(typeName, typeNamespace, declaredType, null);
        }

        public override bool TryResolveType(Type type,
            Type declaredType,
            DataContractResolver knownTypeResolver,
            out XmlDictionaryString typeName,
            out XmlDictionaryString typeNamespace)
        {
            if (m_typeToNames.TryGetValue(type, out Tuple<string, string> value))
            {
                var dictionary = new XmlDictionary();
                typeNamespace = dictionary.Add(value.Item1);
                typeName = dictionary.Add(value.Item2);

                return true;
            }
            else
            {
                return knownTypeResolver.TryResolveType(type, declaredType, null,
                    out typeName, out typeNamespace);
            }
        }

        static string GetNamespace(Type type)
        {
            return type.Namespace ?? DefaultNamespace;
        }

        static string GetName(Type type)
        {
            return type.Name;
        }

        static Type[] ReflectTypes()
        {
            Assembly[] assemblyReferecnes = AppDomain.CurrentDomain.GetAssemblies();

            var types = new List<Type>();

            foreach (Assembly assembly in assemblyReferecnes)
            {
                Type[] typesInReferencedAssembly = GetTypes(assembly);
                types.AddRange(typesInReferencedAssembly);
            }

            return types.ToArray();
        }

        static Type[] GetTypes(Assembly assembly, bool publicOnly = true)
        {
            Type[] allTypes = assembly.GetTypes();

            var types = new List<Type>();

            foreach (Type type in allTypes)
            {
                if (!type.IsEnum &&
                   !type.IsInterface &&
                   !type.IsGenericTypeDefinition)
                {
                    if (publicOnly && !type.IsPublic)
                    {
                        if (!type.IsNested)
                        {
                            continue;
                        }
                        if (type.IsNestedPrivate)
                        {
                            continue;
                        }
                    }
                    types.Add(type);
                }
            }
            return types.ToArray();
        }
    }

    #endregion
}
