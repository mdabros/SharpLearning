using SharpLearning.Containers.Views;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Containers
{
    public static class ArrayExtensions
    {
        /// <summary>
        /// Gets the values from v based on indices
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="v"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public static T[] GetIndices<T>(this T[] v, int[] indices)
        {
            var result = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = v[indices[i]];
            }
            return result;
        }

        /// <summary>
        /// Converts am array of string to an array of floats
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static double[] AsF64(this string[] v)
        {
            return v.Select(s => FloatingPointConversion.ToF64(s)).ToArray();
        }

        /// <summary>
        /// Converts an array of doubles to an array of strings
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static string[] AsString(this double[] v)
        {
            return v.Select(s => FloatingPointConversion.ToString(s)).ToArray();
        }

        /// <summary>
        /// Converts an array of doubles to an array of ints
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static int[] AsInt32(this double[] v)
        {
            return v.Select(s => (int)s).ToArray();
        }

        /// <summary>
        /// Gets a pinned pointer to the double array
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static F64VectorPinnedPtr GetPinnedPointer(this double[] v)
        {
            return new F64VectorPinnedPtr(v);
        }

        /// <summary>
        /// Sorts the keys and values based on the keys within the provided interval 
        /// </summary>
        /// <typeparam name="TKey"></typeparam>
        /// <typeparam name="TValues"></typeparam>
        /// <param name="keys"></param>
        /// <param name="interval"></param>
        /// <param name="values"></param>
        public static void SortWith<TKey, TValues>(this TKey[] keys, Interval1D interval, TValues[] values)
        {
            Array.Sort(keys, values, interval.FromInclusive, interval.Length);
        }

        /// <summary>
        /// Copies the source to the distination within the provided interval
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="interval"></param>
        /// <param name="distination"></param>
        public static void CopyTo<T>(this T[] source, Interval1D interval, T[] distination)
        {
            Array.Copy(source, interval.FromInclusive, distination, interval.FromInclusive, interval.Length);
        }

        /// <summary>
        /// Copies the provided indices from source to destination within the provided interval
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="indices"></param>
        /// <param name="source"></param>
        /// <param name="destination"></param>
        public static void IndexedCopy<T>(this int[] indices, T[] source, Interval1D interval, T[] destination)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                var index = indices[i];
                destination[i] = source[index];
            }
        }

        /// <summary>
        /// Copies the provided indices from source to destination within the provided interval
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="indices"></param>
        /// <param name="source"></param>
        /// <param name="interval"></param>
        /// <param name="destination"></param>
        public static void IndexedCopy(this int[] indices, F64MatrixColumnView source, Interval1D interval, double[] destination)
        {
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                var index = indices[i];
                destination[i] = source[index];
            }
        }

        /// <summary>
        /// Checks if a value is contained in the provided interval of an array 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <param name="interval"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static bool Contains<T>(this T[] array, Interval1D interval, T value)
        {
            var contained = false;
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                if(array[i].Equals(value))
                {
                    contained = true;
                    break;
                }
            }
            return contained;
        }

        /// <summary>
        /// Sums the values within the provided interval
        /// </summary>
        /// <param name="array"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        public static double Sum(this double[] array, Interval1D interval)
        {
            var sum = 0.0;
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                sum += array[i];
            }
            return sum;
        }

        /// <summary>
        /// Shuffles the array in random order
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="list"></param>
        /// <param name="rng"></param>
        public static void Shuffle<T>(this IList<T> list, Random rng)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        /// <summary>
        /// Shuffles the indices then orders them inorder to get the same distribution in each fold. 
        /// </summary>
        /// <typeparam name="TIndex"></typeparam>
        /// <typeparam name="TValues"></typeparam>
        /// <param name="indices"></param>
        /// <param name="values"></param>
        /// <param name="rng"></param>
        /// <param name="folds"></param>
        public static void Stratify<TIndex, TValues>(this TIndex[] indices, IReadOnlyList<TValues> values, Random rng, int folds)
        {
            var zipped = indices.Zip(values, (i, v) => new { Index = i, Value = v }).ToList();
            zipped.Shuffle(rng);

            var grps = zipped.GroupBy(v => v.Value);
            var countsPrFold = grps.Select(g => new { Key = g.Key, CountPrFold = g.Count() / folds }).ToDictionary(v => v.Key, v => v.CountPrFold);

            foreach (var kvp in countsPrFold)
            {
                if (kvp.Value == 0) { throw new ArgumentException("Class: " + kvp.Key + " is too small for " + folds + "-fold statification "); }
            }

            Array.Clear(indices, 0, indices.Length);

            var index = 0;
            for (int i = 0; i < folds; i++)
            {
                foreach (var grp in grps)
                {
                    var counts = countsPrFold[grp.Key];
                    if (i == folds - 1)
                    {
                        var currentIndices = grp.Skip(i * counts).Take(grp.Count()).Select(v => v.Index);
                        foreach (var item in currentIndices)
                        {
                            indices[index++] = item; 
                        }
                    }
                    else
                    {
                        var currentIndices = grp.Skip(i * counts).Take(counts).Select(v => v.Index);
                        foreach (var item in currentIndices)
                        {
                            indices[index++] = item;
                        }
                    }
                }
            }
        }
    }
}
