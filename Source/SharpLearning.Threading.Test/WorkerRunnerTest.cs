using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Concurrent;
using System.Threading;
using System.Diagnostics;
using System.Linq;

namespace SharpLearning.Threading.Test
{
    [TestClass]
    public class WorkerRunnerTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void WorkerRunner_Arguments()
        {
            new ThreadedWorker<int>(0);
        }

        [TestMethod]
        public void WorkerRunner_Run_Processor_Count_1()
        {
            var threads = 1;
            var actual = WorkerRunner_Run(threads);
            
            var expected = new int[] { 893, 894, 877, 887, 889, 888, 886, 884, 879, 890, 887, 883, 888, 893, 892, 883, 890, 880, 880, 894, 894, 882, 881, 886, 890, 887, 887, 881, 884, 888, 891, 883, 890, 890, 895, 884, 889, 879, 882, 890, 890, 892, 887, 887, 893, 889, 893, 894, 889, 882, 882, 885, 883, 890, 888, 884, 893, 893, 882, 891, 879, 889, 884, 891, 889, 880, 887, 889, 888, 881, 886, 880, 890, 882, 888, 889, 890, 884, 891, 887, 892, 885, 889, 884, 886, 879, 879, 884, 887, 889, 876, 885, 882, 887, 884, 880, 875, 889, 890, 891 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void WorkerRunner_Run_Processor_Count_Environment_ProcessorCount()
        {
            var threads = Environment.ProcessorCount;
            var actual = WorkerRunner_Run(threads);

            Array.Sort(actual); // sorts the results inorder to avoid order issues from concurrency
            
            var expected = new int[] { 875, 876, 877, 879, 879, 879, 879, 879, 880, 880, 880, 880, 880, 881, 881, 881, 882, 882, 882, 882, 882, 882, 882, 883, 883, 883, 883, 884, 884, 884, 884, 884, 884, 884, 884, 884, 885, 885, 885, 886, 886, 886, 886, 887, 887, 887, 887, 887, 887, 887, 887, 887, 887, 888, 888, 888, 888, 888, 888, 889, 889, 889, 889, 889, 889, 889, 889, 889, 889, 889, 890, 890, 890, 890, 890, 890, 890, 890, 890, 890, 890, 891, 891, 891, 891, 891, 892, 892, 892, 893, 893, 893, 893, 893, 893, 894, 894, 894, 894, 895 };
            CollectionAssert.AreEqual(expected, actual);
        }

        int[] WorkerRunner_Run(int threads)
        {
           
          
            var random = new Random(42);

            var items = new ConcurrentQueue<int>();
            for (int i = 0; i < 100; i++)
            {
                items.Enqueue(0);
            }

            var strings = new ConcurrentQueue<string>();
            for (int i = 0; i < 100; i++)
            {
                strings.Enqueue(GenerateRandomString(random));
            }

            var testString = GenerateRandomString(random);

            var actions = new List<Action>();
            var results = new ConcurrentBag<int>();
            for (int i = 0; i < threads; i++)
            {
                actions.Add(() => 
                    {
                        int t = -1;
                        while(items.TryDequeue(out t))
                        {
                            string s1;
                            strings.TryDequeue(out s1);

                            var distance = Distance(s1, testString);
                            results.Add(distance);
                        }
                    });
            }

            var sut = new WorkerRunner(actions);

            sut.Run();
            var actual = results.ToArray();
            return actual;
        }

        string GenerateRandomString(Random rand)
        {
            const int length = 1000;
            StringBuilder sb = new StringBuilder(length);
            for (int i = 0; i < length; i++) sb.Append((char)('a' + rand.Next(0, 26)));
            return sb.ToString();
        }

        int Distance(string s1, string s2)
        {
            int[,] dist = new int[s1.Length + 1, s2.Length + 1];
            for (int i = 0; i <= s1.Length; i++) dist[i, 0] = i;
            for (int j = 0; j <= s2.Length; j++) dist[0, j] = j;

            for (int i = 1; i <= s1.Length; i++)
            {
                for (int j = 1; j <= s2.Length; j++)
                {
                    dist[i, j] = (s1[i - 1] == s2[j - 1]) ?
                        dist[i - 1, j - 1] :
                        1 + Math.Min(dist[i - 1, j],
                            Math.Min(dist[i, j - 1],
                                     dist[i - 1, j - 1]));
                }
            }
            return dist[s1.Length, s2.Length];
        }
 
        void Write(int[] results)
        {
            var output = "new int[] {";
            foreach (var result in results)
            {
                output += result + ",";
            }
            Trace.WriteLine(output);
        }
    }
}
