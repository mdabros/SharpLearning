using System;
using System.Linq;

namespace CntkExtensions
{
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
    }
}
