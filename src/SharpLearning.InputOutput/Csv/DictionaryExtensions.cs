using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.InputOutput.Csv
{
    /// <summary>
    /// Extension methods for dictionary
    /// </summary>
    public static class DictionaryExtensions
    {
        /// <summary>
        /// Gets the set of values corresponding to the set of keys
        /// </summary>
        /// <typeparam name="T1"></typeparam>
        /// <typeparam name="T2"></typeparam>
        /// <param name="dictionary"></param>
        /// <param name="keys"></param>
        /// <returns></returns>
        public static T2[] GetValues<T1, T2>(this Dictionary<T1, T2> dictionary, T1[] keys)
        {
            return keys.Select(key => dictionary[key]).ToArray();
        }
    }
}
