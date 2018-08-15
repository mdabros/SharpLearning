using System;
using System.Linq;

namespace CntkExtensions
{
    /// <summary>
    /// Quick and dirty container class for tensor data.
    /// </summary>
    public class Tensor
    {
        public float[] Data;

        /// <summary>
        /// CNTK uses: [W x H x C x N] layout.
        /// W: Widht
        /// H: Height
        /// C: Channels
        /// N: Number of samples
        /// </summary>
        public int[] Dimensions;

        public Tensor(float[] data, params int[] dimensions)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            Dimensions = dimensions ?? throw new ArgumentNullException(nameof(dimensions));

            var totalElements = dimensions.Aggregate((d1, d2) => d1 * d2);
            if(totalElements != data.Length)
            {
                throw new ArgumentNullException($"Data count: {data.Length} does not match " + 
                    $" dimensions [{string.Join(", ", dimensions)}]: {totalElements}");
            }

            Data = data;
            Dimensions = dimensions;
        }

        /// <summary>
        /// Assumes last dimensions is number of samples.
        /// </summary>
        public Tensor GetIndices(params int[] sampleIndices)
        {
            // Assumes last dimensions is number of samples.
            var shape = Dimensions.Take(Dimensions.Length - 1)
                .ToList();

            // Calculate sample size before adding number of samples.
            var sampleSize = shape.Aggregate((d1, d2) => d1 * d2);

            // Add sample count.
            shape.Add(sampleIndices.Length);
            
            var data = new float[shape.Aggregate((d1, d2) => d1 * d2)];
            for (int i = 0; i < sampleIndices.Length; i++)
            {
                var sampleIndex = sampleIndices[i];
                var startIndex = sampleIndex * sampleSize;
                Array.Copy(Data, startIndex, data, i * sampleSize, sampleSize);
            }

            return new Tensor(data, shape.ToArray());
        }
    }
}
