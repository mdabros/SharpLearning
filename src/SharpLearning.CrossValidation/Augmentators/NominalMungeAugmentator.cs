using System;
using System.Collections.Generic;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.CrossValidation.Augmentators
{
    /// <summary>
    /// Augmentates nominal data according to the MUNGE method:
    /// https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf
    /// The method seeks to keep the original distribution of data. This is done by traversing each observation in the dataset
    /// finding its nearest neighbour (euclidean distance) and modifiyng each feature in the observation according to a probability. 
    /// The features are modified using the value from the nearest neighbour as the mean when sampling a new value from a uniform distribution.
    /// </summary>
    public sealed class NominalMungeAugmentator
    {
        readonly double m_probabilityParameter;
        Random m_random;

        /// <summary>
        /// Augmentates nominal data according to the MUNGE method:
        /// https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf
        /// The method seeks to keep the original distribution of data. This is done by traversing each observation in the dataset
        /// finding its nearest neighbour (euclidean distance) and modifiyng each feature in the observation according to a probability. 
        /// The features are modified using the value from the nearest neighbour as the mean when sampling a new value from a uniform distribution.
        /// </summary>
        /// <param name="probabilityParameter">The probability that a feature will be altered with its nearest neighbour. 
        /// Must be between 0.0 and 1.0. (Default is 0.2)</param>
        /// <param name="seed">Seed for random augmentation</param>
        public NominalMungeAugmentator(double probabilityParameter=0.2, int seed = 432)
        {
            if (probabilityParameter > 1.0 || probabilityParameter < 0.0)
            { throw new ArgumentException("probabilityParameter must be between 0.0 and 1.0. Was: " + probabilityParameter); }

            m_probabilityParameter = probabilityParameter;
            m_random = new Random(seed);
        }

        /// <summary>
        /// Returns the agumented version of the data. Excluding the original.
        /// The each feature in the dataset must be scaled/normnalized between 0.0 and 1.0
        /// before the method works.
        /// </summary>
        /// <param name="dataset"></param>
        /// <returns></returns>
        public F64Matrix Agument(F64Matrix dataset)
        {
            var orgCols = dataset.ColumnCount;
            var orgRows = dataset.RowCount;

            var augmentation = new F64Matrix(dataset.RowCount, dataset.ColumnCount);
            var indicesVisited = new HashSet<int>();

            var sample = new double[orgCols];
            var candidate = new double[orgCols];
            indicesVisited.Clear();

            for (int j = 0; j < orgRows; j++)
            {
                if (indicesVisited.Contains(j)) { continue; }
                dataset.Row(j, sample);

                var closestDistance = double.MaxValue;
                var closestIndex = -1;
                indicesVisited.Add(j);

                for (int f = 0; f < orgRows; f++)
                {
                    if (indicesVisited.Contains(f)) { continue; }
                    dataset.Row(f, candidate);

                    var distance = GetHammingDistance(sample, candidate);
                    if(distance < closestDistance)
                    {
                        closestDistance = distance;
                        closestIndex = f;
                    }
                }

                if(closestIndex != -1)
                {
                    dataset.Row(closestIndex, candidate);
                    indicesVisited.Add(closestIndex);

                    for (int h = 0; h < sample.Length; h++)
                    {
                        var sampleValue = sample[h];
                        var candiateValue = candidate[h];

                        if (m_random.NextDouble() <= m_probabilityParameter && m_probabilityParameter != 0.0)
                        {
                            // switch values
                            augmentation.At(j, h, candiateValue);
                            augmentation.At(closestIndex, h, sampleValue);
                        }
                        else 
                        {
                            // keep values
                            augmentation.At(j, h, sampleValue);
                            augmentation.At(closestIndex, h, candiateValue);
                        }
                    }
                }
            }

            return augmentation;
        }
       

        double GetHammingDistance(double[] a, double[] b)
        {
            if (a.Length != b.Length) throw new ArgumentOutOfRangeException("lengths are not equal");
            int count = 0;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i])
                {
                    count++;
                }
            }
            return count;
        }
    }
}
