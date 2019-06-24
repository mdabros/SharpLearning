using System;
using System.Collections.Generic;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.CrossValidation.Augmentators
{
    /// <summary>
    /// Augmentates continuous data according to the MUNGE method:
    /// https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf
    /// The method seeks to keep the original distribution of data. This is done by traversing each observation in the dataset
    /// finding its nearest neighbour (euclidean distance) and modifiyng each feature in the observation according to a probability. 
    /// The features are modified using the value from the nearest neighbour as the mean when sampling a new value from a uniform distribution.
    /// </summary>
    public sealed class ContinuousMungeAugmentator
    {
        readonly double m_probabilityParameter;
        readonly double m_localVariance;
        Random m_random;

        /// <summary>
        /// Augmentates continuous data according to the MUNGE method:
        /// https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf
        /// The method seeks to keep the original distribution of data. This is done by traversing each observation in the dataset
        /// finding its nearest neighbour (euclidean distance) and modifiyng each feature in the observation according to a probability. 
        /// The features are modified using the value from the nearest neighbour as the mean when sampling a new value from a uniform distribution.
        /// </summary>
        /// <param name="probabilityParameter">The probability that a feature will be altered with its nearest neighbour. 
        /// Must be between 0.0 and 1.0. (Default is 0.2)</param>
        /// <param name="localVariance">Variance when sampling a new value for an augmentated sample. (Default is 1.0)</param>
        /// <param name="seed">Seed for random augmentation</param>
        public ContinuousMungeAugmentator(double probabilityParameter=0.2, double localVariance=1.0, int seed = 432)
        {
            if (probabilityParameter > 1.0 || probabilityParameter < 0.0)
            { throw new ArgumentException("probabilityParameter must be between 0.0 and 1.0. Was: " + probabilityParameter); }

            m_probabilityParameter = probabilityParameter;
            m_localVariance = localVariance;

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

                    var distance = GetDistance(sample, candidate);
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
                            var std = (sampleValue - candiateValue) / m_localVariance;

                            augmentation.At(j, h, SampleRandom(candiateValue, std));
                            augmentation.At(closestIndex, h, SampleRandom(sampleValue, std));
                        }
                        else // keep values
                        {
                            augmentation.At(j, h, sampleValue);
                            augmentation.At(closestIndex, h, candiateValue);
                        }
                    }
                }
            }

            return augmentation;
        }

        double SampleRandom(double mean, double std)
        {
            double u1 = m_random.NextDouble(); //these are uniform(0,1) random doubles
            double u2 = m_random.NextDouble();

            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); // random normal(0,1)
            double randNormal = mean + std * randStdNormal; //random normal(mean,stdDev^2)

            return randNormal;
        }
        
        double GetDistance(double[] p, double[] q)
        {
            double distance = 0;
            double diff = 0;

            if (p.Length != q.Length)
                throw new ArgumentException("Input vectors must be of the same dimension.");

            for (int x = 0, len = p.Length; x < len; x++)
            {
                diff = p[x] - q[x];
                distance += diff * diff;
            }

            return distance;//Math.Sqrt(distance);
        }
    }
}
