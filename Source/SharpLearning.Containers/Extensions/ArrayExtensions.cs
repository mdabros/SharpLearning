using SharpLearning.Containers.Views;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Containers.Extensions
{
    public static class ArrayExtensions
    {
        /// <summary>
        /// Converts Nan to 0.0, NegativeInfinity to double.MinValue and PositiveInfinity to double.MaxValue
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double NanToNum(this double value)
        {
            if (double.IsNaN(value))
            {
                value = 0.0;
            }

            if (double.IsNegativeInfinity(value))
            {
                value = double.MinValue;
            }

            if (double.IsPositiveInfinity(value))
            {
                value = double.MaxValue;
            }

            return value;
        }

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
        /// Sorts the keys and values based on the keys
        /// </summary>
        /// <typeparam name="TKey"></typeparam>
        /// <typeparam name="TValues"></typeparam>
        /// <param name="keys"></param>
        /// <param name="interval"></param>
        /// <param name="values"></param>
        public static void SortWith<TKey, TValues>(this TKey[] keys, TValues[] values)
        {
            Array.Sort(keys, values, 0, keys.Length);
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
        /// Copies the provided indices from source to destination
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="indices"></param>
        /// <param name="source"></param>
        /// <param name="interval"></param>
        /// <param name="destination"></param>
        public static void IndexedCopy<T>(this int[] indices, T[] source, T[] destination)
        {
            for (int i = 0; i < indices.Length; i++)
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
        /// Sums the values given by the indices
        /// </summary>
        /// <param name="array"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public static double Sum(this double[] array, int[] indices)
        {
            var sum = 0.0;
            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                sum += array[index];
            }
            return sum;
        }

        /// <summary>
        /// Returns the exponential indexed value from each array in the list
        /// </summary>
        /// <param name="array"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        public static double[] Exp(this List<double[]> array, int index)
        {
            var exp = new double[array.Count];
            for (int i = 0; i < array.Count; i++)
            {
                exp[i] = Math.Exp(array[i][index]);
            }
            return exp;
        }

        /// <summary>
        /// Returns the cummulative sum of an array
        /// </summary>
        /// <param name="array"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public static double[] CumSum(this double[] array)
        {
            var sum = 0.0;
            var cumsum = new double[array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                sum += array[i];
                cumsum[i] = sum;
            }

            return cumsum;
        }

        /// <summary>
        /// Calculates the weighted mean
        /// </summary>
        /// <param name="array"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public static double WeightedMean(this double[] array, double[] weights)
        {
            var mean = 0.0;
            for (int i = 0; i < array.Length; i++)
            {
                mean += array[i] * weights[i];
            }

            mean = mean / weights.Sum();

            return mean;
        }

        /// <summary>
        /// Calculates the weighted median. Expects values and weights to be sorted according to values
        /// http://stackoverflow.com/questions/9794558/weighted-median-computation
        /// </summary>
        /// <param name="array"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public static double WeightedMedian(this double[] values, double[] weights)
        {
            double total = weights.Sum(); // the total weight

            int k = 0;
            double sum = total - weights[0]; // sum is the total weight of all `x[i] > x[k]`

            while (sum > total / 2)
            {
                ++k;
                sum -= weights[k];
            }

            return values[k];
        }

        /// <summary>
        /// Calculates the median from the provided indices.
        /// </summary>
        /// <param name="values"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public static double Median(this double[] values, int[] indices)
        {
            var array = new double[indices.Length];
            indices.IndexedCopy(values, array);
            Array.Sort(array);

            if (indices.Length % 2 == 0)
            {
                var index1 = (int)((indices.Length / 2.0) - 0.5);
                var v1 = array[index1];

                var index2 = (int)((indices.Length / 2.0) + 0.5);
                var v2 = array[index2];

                return (v1 + v2) / 2.0;
            }

            return array[indices.Length / 2];
        }

        /// <summary>
        /// Calculates the median
        /// </summary>
        /// <param name="values"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public static double Median(this double[] values)
        {
            var array = new double[values.Length];
            Array.Copy(values, array, values.Length);
            Array.Sort(array);

            if (array.Length % 2 == 0)
            {
                var index1 = (int)((array.Length / 2.0) - 0.5);
                var v1 = array[index1];

                var index2 = (int)((array.Length / 2.0) + 0.5);
                var v2 = array[index2];

                return (v1 + v2) / 2.0;
            }

            return array[array.Length / 2];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="values"></param>
        /// <param name="percentile"></param>
        /// <returns></returns>
        public static double ScoreAtPercentile(this double[] values, double percentile)
        {
            if (percentile == 1.0)
                return values.Max();

            if (percentile == 0.0)
                return values.Min();

            var array = new double[values.Length];
            Array.Copy(values, array, values.Length);
            //indices.IndexedCopy(values, array);
            Array.Sort(array);

            var index = percentile * (values.Length - 1.0);
            var i = (int)index;
            var diff = index - i;
            
            if(diff != 0.0)
            {
                var j = i + 1;
                var v1 = array[i];
                var w1 = j - index;

                var v2 = array[j];
                var w2 = index - i;
                
                return (v1 * w1 + v2 * w2) / (w1 + w2);
            }

            return array[i];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="values"></param>
        /// <param name="percentile"></param>
        /// <returns></returns>
        public static double ScoreAtPercentile(this double[] values, double percentile, int[] indices)
        {
            var array = new double[indices.Length];
            indices.IndexedCopy(values, array);

            if (percentile == 1.0)
                return array.Max();

            if (percentile == 0.0)
                return array.Min();

            Array.Sort(array);

            var index = percentile * (array.Length - 1.0);
            var i = (int)index;
            var diff = index - i;

            if (diff != 0.0)
            {
                var j = i + 1;
                var v1 = array[i];
                var w1 = j - index;

                var v2 = array[j];
                var w2 = index - i;

                return (v1 * w1 + v2 * w2) / (w1 + w2);
            }

            return array[i];
        }

        /// <summary>
        /// Calculates the weighted mean from the indices
        /// </summary>
        /// <param name="array"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public static double WeightedMean(this double[] array, double[] weights, int[] indices)
        {
            var mean = 0.0;
            var wSum = 0.0;
            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                var w = weights[index];
                wSum += w;

                mean += array[index] * w;
            }

            mean = mean / wSum;

            return mean;
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
        public static void Stratify<TValues>(this int[] indices, IReadOnlyList<TValues> values, Random rng, int folds)
        {
            if (indices.Length != values.Count)
            {
                throw new ArgumentException("Indices length: " + indices.Length + " differs from values length: " + values.Count);
            }

            var zipped = indices.Zip(values, (i, v) => new { Index = i, Value = v }).ToList();
            zipped.Shuffle(rng);

            var grps = zipped.GroupBy(v => v.Value);
            var countsPrFold = grps.Select(g => new { Key = g.Key, CountPrFold = g.Count() / folds }).ToDictionary(v => v.Key, v => v.CountPrFold);

            foreach (var kvp in countsPrFold)
            {
                if (kvp.Value == 0) { throw new ArgumentException("Class: " + kvp.Key + " Count: " + kvp.Value + " is too small for " + folds + "-fold statification "); }
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

        /// <summary>
        /// Takes a stratified sample of size sampleSize with distributions equal to the input data.
        /// http://en.wikipedia.org/wiki/Stratified_sampling
        /// Returns a set of indices corresponding to the samples chosen. 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="data"></param>
        /// <param name="sampleSize"></param>
        /// <returns></returns>
        public static int[] StratifiedIndexSampling<T>(this T[] data, int sampleSize)
        {
            if (data.Length < sampleSize)
            { throw new ArgumentException("SampleSize " + sampleSize + " is larger than data size " + data.Length); }

            var RequiredSamples = data.GroupBy(d => d)
                .ToDictionary(d => d.Key, d => Math.Round((double)d.Count() / (double)data.Length * (double)sampleSize));

            foreach (var kvp in RequiredSamples)
            {
                if (kvp.Value == 0) { throw new ArgumentException("Sample size is too small for value: " + kvp.Key + " to be included."); }
            }

            var currentSampleCount = RequiredSamples.ToDictionary(k => k.Key, k => 0);
            var sampleIndices = new int[sampleSize];
            var sampleIndex = 0;
                        
            for (int i = 0; i < data.Length; i++)
            {
                var value = data[i];
                if(currentSampleCount[value] != RequiredSamples[value])
                {
                    sampleIndices[sampleIndex++] = i;
                    currentSampleCount[value]++;
                }
            }

            return sampleIndices;
        }

        /// <summary>
        /// Takes a stratified sample of size sampleSize with distributions equal to the input data.
        /// http://en.wikipedia.org/wiki/Stratified_sampling
        /// Returns a set of indices corresponding to the samples chosen. 
        /// Only samples within the indies provided in dataIndices
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="data"></param>
        /// <param name="sampleSize"></param>
        /// <param name="dataIndices"></param>
        /// <returns></returns>
        public static int[] StratifiedIndexSampling<T>(this T[] data, int sampleSize, int[] dataIndices)
        {
            if (dataIndices.Length < sampleSize) 
            { throw new ArgumentException("SampleSize " + sampleSize + " is larger than dataIndices size " + dataIndices.Length); }
            if (data.Length < dataIndices.Length) 
            { throw new ArgumentException("dataIndices " + dataIndices.Length + " is larger than data size " + data.Length); }

            var RequiredSamples = data.GroupBy(d => d)
                .ToDictionary(d => d.Key, d => Math.Round((double)d.Count() / (double)data.Length * (double)sampleSize));

            foreach (var kvp in RequiredSamples)
            {
                if (kvp.Value == 0) { throw new ArgumentException("Sample size is too small for value: " + kvp.Key + " to be included."); }
            }

            var currentSampleCount = RequiredSamples.ToDictionary(k => k.Key, k => 0);
            var sampleIndices = new int[sampleSize];
            var sampleIndex = 0;

            for (int i = 0; i < dataIndices.Length; i++)
            {
                var dataIndex = dataIndices[i];
                var value = data[dataIndex];
                if (currentSampleCount[value] != RequiredSamples[value])
                {
                    sampleIndices[sampleIndex++] = dataIndex;
                    currentSampleCount[value]++;
                }
            }

            return sampleIndices;
        }

    }
}
