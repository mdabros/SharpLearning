using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;

namespace SharpLearning.Containers.Extensions
{
    /// <summary>
    /// 
    /// </summary>
    public static class ArrayExtensions
    {
        /// <summary>
        /// 
        /// </summary>
        public static readonly Converter<string, double> DefaultF64Converter = FloatingPointConversion.ToF64;

        /// <summary>
        /// Clears array.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        public static void Clear<T>(this T[] array)
        {
            Array.Clear(array, 0, array.Length);
        }

        /// <summary>
        /// Iterates over an array and perform the action at each element
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <param name="a"></param>
        public static void ForEach<T>(this T[] array, Action<T> a)
        {
            foreach (var value in array)
            {
                a(value);
            }
        }

        /// <summary>
        /// Iterates over an array and applies the function a to the elements.
        /// The values are updated directly in the array.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <param name="a"></param>
        public static void Map<T>(this T[] array, Func<T> a)
        {
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = a();
            }
        }

        /// <summary>
        /// Iterates over an array and applies the function a to the elements.
        /// The values are updated directly in the array.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="array"></param>
        /// <param name="a"></param>
        public static void Map<T>(this T[] array, Func<T, T> a)
        {
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = a(array[i]);
            }
        }


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
        /// Converts an array of string to an array of floats
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        public static double[] AsF64(this string[] v)
        {
            return AsF64(v, DefaultF64Converter);
        }

        /// <summary>
        /// /// Converts an array of string to an array of floats
        /// </summary>
        /// <param name="v"></param>
        /// <param name="converter"></param>
        /// <returns></returns>
        public static double[] AsF64(this string[] v, Converter<string, double> converter)
        {
            return v.Select(s => converter(s)).ToArray();
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
        /// Copies the source to the destination within the provided interval
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="source"></param>
        /// <param name="interval"></param>
        /// <param name="destination"></param>
        public static void CopyTo<T>(this T[] source, Interval1D interval, T[] destination)
        {
            Array.Copy(source, interval.FromInclusive, destination, interval.FromInclusive, interval.Length);
        }

        /// <summary>
        /// Copies the provided indices from source to destination within the provided interval
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="indices"></param>
        /// <param name="source"></param>
        /// <param name="interval"></param>
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
        /// <param name="indices"></param>
        /// <param name="source"></param>
        /// <param name="interval"></param>
        /// <param name="destination"></param>
        public static void IndexedCopy(this int[] indices, F64MatrixColumnView source,
            Interval1D interval, double[] destination)
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
        /// Calculates the weighted median. Expects values and weights to be sorted according to values
        /// http://stackoverflow.com/questions/9794558/weighted-median-computation
        /// </summary>
        /// <param name="values"></param>
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
        /// Calculates the median
        /// </summary>
        /// <param name="values"></param>
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
        /// Converts an array of arrays to an F64Matrix
        /// </summary>
        /// <param name="m"></param>
        /// <returns></returns>
        public static F64Matrix ToF64Matrix(this double[][] m)
        {
            return ToF64Matrix(m.ToList());
        }

        /// <summary>
        /// Converts a list of arrays to an F64Matrix
        /// </summary>
        /// <param name="m"></param>
        /// <returns></returns>
        public static F64Matrix ToF64Matrix(this List<double[]> m)
        {
            var rows = m.Count;
            var cols = m.First().Length;

            var matrix = new F64Matrix(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                var row = m[i];
                if (row.Length != cols)
                {
                    throw new ArgumentException("Conversion to F64Matrix requires all row to be equal length");
                }

                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = row[j];
                }
            }

            return matrix;
        }

        /// <summary>
        /// Calculates the weighted mean from the indices
        /// </summary>
        /// <param name="array"></param>
        /// <param name="weights"></param>
        /// <param name="indices"></param>
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
        /// <param name="random"></param>
        public static void Shuffle<T>(this IList<T> list, Random random)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = random.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
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
        /// <param name="random"></param>
        /// <returns></returns>
        public static int[] StratifiedIndexSampling<T>(this T[] data, int sampleSize, Random random)
        {
            if (data.Length < sampleSize)
            {
                throw new ArgumentException("SampleSize " + sampleSize +
                    " is larger than data size " + data.Length);
            }

            var requiredSamples = data.GroupBy(d => d)
                .ToDictionary(d => d.Key, d => (int)Math.Round((double)d.Count() / data.Length * sampleSize));

            foreach (var kvp in requiredSamples)
            {
                if (kvp.Value == 0)
                {
                    throw new ArgumentException("Sample size is too small for value: " +
                        kvp.Key + " to be included.");
                }
            }

            // Shuffle the indices to avoid sampling the data in original order.
            var indices = Enumerable.Range(0, data.Length).ToArray();
            indices.Shuffle(random);

            var currentSampleCount = requiredSamples.ToDictionary(k => k.Key, k => 0);

            // might be slightly different than the specified depending on data distribution
            var actualSampleSize = requiredSamples.Select(s => s.Value).Sum();

            // if actual sample size is different from specified add/subtract duff from largest class
            if (actualSampleSize != sampleSize)
            {
                var diff = sampleSize - actualSampleSize;
                var largestClassKey = requiredSamples.OrderByDescending(s => s.Value).First().Key;
                requiredSamples[largestClassKey] += diff;
            }

            var sampleIndices = new int[sampleSize];
            var sampleIndex = 0;

            for (int i = 0; i < data.Length; i++)
            {
                var index = indices[i];
                var value = data[index];
                if (currentSampleCount[value] != requiredSamples[value])
                {
                    sampleIndices[sampleIndex++] = index;
                    currentSampleCount[value]++;
                }

                if (sampleIndex == sampleSize)
                {
                    break;
                }
            }

            if (requiredSamples.Select(s => s.Value).Sum() != sampleSize)
            {
                throw new ArgumentException("Actual sample size: " + actualSampleSize +
                    " is different than specified sample size: " + sampleSize);
            }

            return sampleIndices;
        }

        /// <summary>
        /// Takes a stratified sample of size sampleSize with distributions equal to the input data.
        /// http://en.wikipedia.org/wiki/Stratified_sampling
        /// Returns a set of indices corresponding to the samples chosen. 
        /// Only samples within the indices provided in dataIndices
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="data"></param>
        /// <param name="sampleSize"></param>
        /// <param name="dataIndices"></param>
        /// <param name="random"></param>
        /// <returns></returns>
        public static int[] StratifiedIndexSampling<T>(this T[] data, int sampleSize, int[] dataIndices, Random random)
        {
            if (dataIndices.Length < sampleSize)
            {
                throw new ArgumentException("SampleSize " + sampleSize +
                    " is larger than dataIndices size " + dataIndices.Length);
            }

            if (data.Length < dataIndices.Length)
            {
                throw new ArgumentException("dataIndices " + dataIndices.Length +
                    " is larger than data size " + data.Length);
            }

            var requiredSamples = data.GroupBy(d => d)
                .ToDictionary(d => d.Key, d => (int)Math.Round((double)d.Count() / data.Length * sampleSize));

            foreach (var kvp in requiredSamples)
            {
                if (kvp.Value == 0)
                {
                    throw new ArgumentException("Sample size is too small for value: " +
                        kvp.Key + " to be included.");
                }
            }

            var currentSampleCount = requiredSamples.ToDictionary(k => k.Key, k => 0);
            // might be slightly different than the specified depending on data distribution
            var actualSampleSize = requiredSamples.Select(s => s.Value).Sum();

            // if actual sample size is different from specified add/subtract difference from largest class
            if (actualSampleSize != sampleSize)
            {
                var diff = sampleSize - actualSampleSize;
                var largestClassKey = requiredSamples.OrderByDescending(s => s.Value).First().Key;
                requiredSamples[largestClassKey] += diff;
            }

            var sampleIndices = new int[sampleSize];
            var sampleIndex = 0;

            // Shuffle the indices to avoid sampling the data in original order.
            var indices = dataIndices.ToArray();
            indices.Shuffle(random);

            for (int i = 0; i < indices.Length; i++)
            {
                var dataIndex = indices[i];
                var value = data[dataIndex];
                if (currentSampleCount[value] != requiredSamples[value])
                {
                    sampleIndices[sampleIndex++] = dataIndex;
                    currentSampleCount[value]++;
                }

                if (sampleIndex == sampleSize)
                {
                    break;
                }
            }

            if (requiredSamples.Select(s => s.Value).Sum() != sampleSize)
            {
                throw new ArgumentException("Actual sample size: " + actualSampleSize +
                    " is different than specified sample size: " + sampleSize);
            }

            return sampleIndices;
        }
    }
}
